"""
SCAFFOLD-CF: causally-constrained counterfactual recourse.

Structurally Constrained Actionable Filtering For On-manifold Low-cost
Decisions. The method enumerates a library of candidate interventions,
resolves each one so that the consequences entailed by a declared rule set are
realised, and then filters rather than penalises: a candidate is retained only
if it flips the classifier, touches only actionable features, and leaves no
declared rule unsatisfied. Survivors are tightened and reduced to a small
returned set by Pareto selection. When nothing survives the causal gate the
method abstains.

The causal rule set is a *declared* input, not a validated causal model.
Conditions C1 and C2 below are properties of a candidate relative to that rule
set; they are not claims about what an employer would observe after
intervening.

Extracted from the benchmark notebooks accompanying the paper. The algorithm
bodies are unchanged; dataset-specific globals have been lifted into
``FeatureSpec`` and ``CausalRule`` so the method can be applied to other data.

Usage
-----
    spec = FeatureSpec(
        features=list(X_train.columns),
        limits={f: (X_train[f].min(), X_train[f].max()) for f in X_train},
        immutable=["age", "gender", "education"],
        downstream_only=["family_supports", "mentally_well"],
    )
    rules = [
        CausalRule("working_hour", "work_life_balance", direction=-1, threshold=1),
        CausalRule("workload_satisfaction", "mentally_well", direction=+1, threshold=2),
    ]
    explainer = ScaffoldCF(model, X_train, spec, rules, desired_class=0).fit()
    cfs = explainer.explain(x)          # list of arrays; [] means abstention

Licence: see LICENSE-CODE.
"""

from __future__ import annotations

import itertools
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

__all__ = ["FeatureSpec", "CausalRule", "ScaffoldCF"]


# ---------------------------------------------------------------------------
# Dataset description
# ---------------------------------------------------------------------------

@dataclass
class FeatureSpec:
    """Feature roles and bounds.

    ``immutable`` features cannot be intervened on at all. ``downstream_only``
    features are states an actor cannot set directly; they may change as a
    consequence of a rule firing, but never as a deliberate intervention.
    Everything else is mutable.
    """

    features: List[str]
    limits: Dict[str, Tuple[float, float]]
    immutable: List[str] = field(default_factory=list)
    downstream_only: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        blocked = set(self.immutable) | set(self.downstream_only)
        self.mutable = [f for f in self.features if f not in blocked]
        self.index = {f: j for j, f in enumerate(self.features)}
        self.ranges = np.array(
            [max(self.limits[f][1] - self.limits[f][0], 1e-12) for f in self.features],
            dtype=float,
        )


@dataclass(frozen=True)
class CausalRule:
    """A monotone floor: moving ``cause`` entails a minimum move in ``effect``.

    The rule fires when ``direction * (z[cause] - x[cause]) >= threshold``, so
    ``direction=-1`` encodes "when the cause falls" and ``+1`` "when it rises".
    On firing, ``effect`` may not remain below ``x[effect] + step``, bounded by
    the effect's observed range. Rules are floors, never equalities: a
    candidate that raises the effect further is still admissible.
    """

    cause: str
    effect: str
    direction: int = 1
    threshold: float = 1.0
    step: float = 1.0


# ---------------------------------------------------------------------------
# The explainer
# ---------------------------------------------------------------------------

class ScaffoldCF:
    """Causally-constrained counterfactual explainer.

    Parameters
    ----------
    model
        Any fitted classifier exposing ``predict``, ``predict_proba`` and
        ``classes_``. No differentiability is required.
    X_train
        Training feature frame, used for the intervention library, the value
        support, the density model and the plausibility reference.
    spec, rules
        Dataset description and the declared rule set.
    desired_class
        The favourable class a counterfactual must reach.
    k
        Number of counterfactuals returned per query.
    max_bundle_features
        Interpretability cap on how many features one intervention may touch.
    n_lib_neighbors
        Favourable-class neighbours mined per unfavourable training row.
    refine_n
        Sparsest pool members passed to greedy tightening.
    k_p
        Neighbours averaged for the target-proximity objective.
    j_smooth
        Perturbed query copies used to smooth the selection objectives.
    tau_offset, tau_lo, tau_hi
        Adaptive confidence margin: ``clip(p(y'|x) + tau_offset, tau_lo, tau_hi)``.
    lof_neighbors, lof_quantile
        Density model and the percentile of real favourable-class scores used
        as the admission cutoff.
    eps
        Minimum Gower separation between returned counterfactuals.
    relax_causal_on_empty
        If True, restores the causal-relaxation fallback present in the
        original notebooks, which dropped C1 and C2 when the gated pool was
        empty. It is off by default because the published results are
        unaffected by it: auditing every query-seed across five seeds, the
        relaxation was reached on 20 of 196 turnover-intention queries and on
        every occasion the relaxed pool was empty as well, so the method
        abstained rather than returning a relaxed candidate. Leaving it off
        makes the causal guarantee unconditional, which is what the paper
        claims.
    """

    def __init__(
        self,
        model,
        X_train,
        spec: FeatureSpec,
        rules: Sequence[CausalRule],
        desired_class,
        *,
        k: int = 3,
        max_bundle_features: int = 3,
        n_lib_neighbors: int = 5,
        refine_n: int = 8,
        k_p: int = 5,
        j_smooth: int = 3,
        tau_offset: float = 0.20,
        tau_lo: float = 0.52,
        tau_hi: float = 0.60,
        lof_neighbors: int = 15,
        lof_quantile: float = 0.05,
        eps: float = 0.02,
        seed: int = 42,
        relax_causal_on_empty: bool = False,
    ) -> None:
        self.model = model
        self.X_train = X_train
        self.spec = spec
        self.rules = list(rules)
        self.desired = desired_class
        self.k = k
        self.max_bundle_features = max_bundle_features
        self.n_lib_neighbors = n_lib_neighbors
        self.refine_n = refine_n
        self.k_p = k_p
        self.j_smooth = j_smooth
        self.tau_offset, self.tau_lo, self.tau_hi = tau_offset, tau_lo, tau_hi
        self.lof_neighbors, self.lof_quantile = lof_neighbors, lof_quantile
        self.eps = eps
        self.seed = seed
        self.relax_causal_on_empty = relax_causal_on_empty

        self._validate()

    def _validate(self) -> None:
        known = set(self.spec.features)
        for r in self.rules:
            for role, f in (("cause", r.cause), ("effect", r.effect)):
                if f not in known:
                    raise ValueError(f"rule {role} {f!r} is not a feature")
            if r.direction not in (-1, 1):
                raise ValueError("CausalRule.direction must be -1 or +1")
            if r.effect in self.spec.immutable:
                raise ValueError(f"rule effect {r.effect!r} is immutable")

    # -- fitting ------------------------------------------------------------

    def fit(self) -> "ScaffoldCF":
        """Build the value support, intervention library and density model."""
        sp, X = self.spec, self.X_train
        self._support = {f: np.sort(X[f].unique()) for f in sp.features}
        self._tgt_col = list(self.model.classes_).index(self.desired)

        preds = self.model.predict(X.values)
        self._fav = X[preds == self.desired]
        unfav = X[preds != self.desired]
        if len(self._fav) == 0:
            raise ValueError("no training row is predicted as the desired class")

        self._library = self._build_library(unfav)

        favn = self._fav.values / sp.ranges
        self._lof = LocalOutlierFactor(
            n_neighbors=min(self.lof_neighbors, max(2, len(favn) - 1)),
            novelty=True,
            metric="manhattan",
        ).fit(favn)
        self._lof_cutoff = float(
            np.quantile(self._lof.score_samples(favn), self.lof_quantile)
        )

        self._search_cols = [
            sp.index[f] for f in sp.features
            if f in sp.mutable or f in sp.downstream_only
        ]
        return self

    def _build_library(self, unfav) -> List[Dict[str, int]]:
        """Mined transitions plus enumerated edits, deduplicated and capped."""
        sp = self.spec
        mined: List[Dict[str, int]] = []
        if len(unfav):
            nn = NearestNeighbors(
                n_neighbors=min(self.n_lib_neighbors, len(self._fav))
            ).fit(self._fav.values / sp.ranges)
            for row in unfav.values.astype(float):
                _, idx = nn.kneighbors((row / sp.ranges).reshape(1, -1))
                for si in idx[0]:
                    fav = self._fav.iloc[si]
                    bundle = {
                        f: int(round(fav[f] - row[sp.index[f]]))
                        for f in sp.mutable
                        if int(round(fav[f] - row[sp.index[f]])) != 0
                    }
                    if bundle:
                        mined.append(bundle)

        mags = {f: int(round(sp.limits[f][1] - sp.limits[f][0])) for f in sp.mutable}
        synth: List[Dict[str, int]] = []
        for f in sp.mutable:                                     # atomic
            for m in range(1, mags[f] + 1):
                synth.append({f: +m})
                synth.append({f: -m})
        for f1, f2 in itertools.combinations(sp.mutable, 2):     # pairwise
            for m1 in range(1, mags[f1] + 1):
                for m2 in range(1, mags[f2] + 1):
                    for s1 in (+1, -1):
                        for s2 in (+1, -1):
                            synth.append({f1: s1 * m1, f2: s2 * m2})
        for f1, f2, f3 in itertools.combinations(sp.mutable, 3):  # triples
            for s1 in (+1, -1):
                for s2 in (+1, -1):
                    for s3 in (+1, -1):
                        synth.append({f1: s1, f2: s2, f3: s3})

        seen, merged = set(), []
        for b in synth + mined:
            key = tuple(sorted(b.items()))
            if key not in seen:
                seen.add(key)
                merged.append(b)
        kept = [b for b in merged if len(b) <= self.max_bundle_features]
        return sorted(kept, key=lambda b: (len(b), sum(abs(v) for v in b.values())))

    # -- resolution operator ------------------------------------------------

    def propagate(self, xv: np.ndarray, zv: np.ndarray) -> np.ndarray:
        """Lift each rule's effect to its floor, jointly and transitively.

        Run to a fixed point over three passes so that chained rules fire: a
        cause moving lifts its effect, which may in turn be the cause of
        another rule.
        """
        sp, z = self.spec, zv.copy()
        for _ in range(3):
            for r in self.rules:
                jc, je = sp.index[r.cause], sp.index[r.effect]
                if r.direction * (z[jc] - xv[jc]) >= r.threshold:
                    z[je] = max(z[je], min(sp.limits[r.effect][1], xv[je] + r.step))
        return z

    def causal_violations(self, xv: np.ndarray, zv: np.ndarray) -> int:
        """Count rules whose effect is left short of its implied floor.

        Applicable to any method's output, which is what makes the causal
        comparison across explainers an equal one.
        """
        sp, v = self.spec, 0
        for r in self.rules:
            jc, je = sp.index[r.cause], sp.index[r.effect]
            if r.direction * (zv[jc] - xv[jc]) >= r.threshold:
                if zv[je] < min(sp.limits[r.effect][1], xv[je] + r.step):
                    v += 1
        return v

    def snap_to_support(self, Z: np.ndarray) -> np.ndarray:
        """Force every coordinate onto a value that occurs in the data."""
        Z = Z.copy()
        for f, j in self.spec.index.items():
            vals = self._support[f]
            Z[:, j] = vals[np.abs(vals[None, :] - Z[:, j][:, None]).argmin(1)]
        return Z

    def _apply(self, xv: np.ndarray, bundles: Sequence[Dict[str, int]]) -> np.ndarray:
        sp = self.spec
        Z = np.tile(xv.astype(float), (len(bundles), 1))
        for i, b in enumerate(bundles):
            for f, d in b.items():
                j = sp.index[f]
                lo, hi = sp.limits[f]
                Z[i, j] = np.clip(xv[j] + d, lo, hi)
        return Z

    def _resolve(self, xv, bundles, use_causal=True, use_snap=True) -> np.ndarray:
        """Phi(x, b) = Snap(Propagate(Clip(x + b))), batched."""
        if not bundles:
            return np.empty((0, len(xv)))
        Z = self._apply(xv, bundles)
        if use_causal:
            Z = np.vstack([self.propagate(xv, z) for z in Z])
        if use_snap:
            Z = self.snap_to_support(Z)
        return Z

    # -- objectives ---------------------------------------------------------

    def _gower(self, xv, zv) -> float:
        return float(np.mean(np.abs(xv - zv) / self.spec.ranges))

    @staticmethod
    def _l0(xv, zv) -> int:
        return int(np.sum(xv != zv))

    def _implaus(self, zv) -> float:
        Xr = self.X_train.values[:, self._search_cols]
        rg = self.spec.ranges[self._search_cols]
        return float(np.min(np.linalg.norm((Xr - zv[self._search_cols]) / rg, axis=1)))

    def _p_target(self, zv) -> float:
        d = np.mean(np.abs(self._fav.values - zv) / self.spec.ranges, axis=1)
        return float(1.0 - np.sort(d)[: self.k_p].mean())

    def _p_class(self, Z) -> np.ndarray:
        return self.model.predict_proba(Z)[:, self._tgt_col]

    # -- gates --------------------------------------------------------------

    def _actionable(self, bundle: Dict[str, int]) -> bool:
        """An intervention is actionable iff it touches mutable features only."""
        blocked = set(self.spec.immutable) | set(self.spec.downstream_only)
        return not any(d != 0 and f in blocked for f, d in bundle.items())

    def _consequence_ok(self, xv, zv, bundle) -> bool:
        """C1: every changed coordinate is intervened on or rule-entailed."""
        expected = self.propagate(xv, self._apply(xv, [bundle])[0])
        for j, f in enumerate(self.spec.features):
            if zv[j] != xv[j] and f not in bundle and zv[j] != expected[j]:
                return False
        return True

    def _gate_ok(self, xv, bundle, zv) -> bool:
        """Actionability, C1 consequence discipline, C2 causal completeness."""
        return (
            self._actionable(bundle)
            and self._consequence_ok(xv, zv, bundle)
            and self.causal_violations(xv, zv) == 0
        )

    def _tau(self, xv) -> float:
        p = float(self.model.predict_proba(xv.reshape(1, -1))[0, self._tgt_col])
        return float(np.clip(p + self.tau_offset, self.tau_lo, self.tau_hi))

    def _dense(self, zv) -> bool:
        """Admit only candidates in populated regions of the feature space.

        ``score_samples`` returns the *negated* local outlier factor, so larger
        values are more typical and the comparison is ``>=``.
        """
        s = self._lof.score_samples((zv / self.spec.ranges).reshape(1, -1))[0]
        return s >= self._lof_cutoff

    # -- pool ---------------------------------------------------------------

    def candidate_pool(self, xv, use_causal=True, use_snap=True, tau=None):
        """Resolve the library and apply the three filters in order.

        Returns ``(bundle, candidate)`` pairs, or an empty list when the method
        abstains. Causal admissibility is never relaxed unless the explainer
        was constructed with ``relax_causal_on_empty=True``; the density and
        margin fallbacks select among candidates that have already passed the
        causal gate, so neither affects C1 or C2.
        """
        tau_target = self._tau(xv) if tau is None else tau
        lib = self._library
        Z = self._resolve(xv, lib, use_causal, use_snap)
        p = self._p_class(Z)

        # Stage 1 -- validity, actionability, causal admissibility.
        gated = [i for i in np.flatnonzero(p > 0.5 + 1e-9)
                 if self._gate_ok(xv, lib[i], Z[i])]

        if not gated and self.relax_causal_on_empty:
            gated = [i for i in np.flatnonzero(p > 0.5 + 1e-9)
                     if self._actionable(lib[i])]
        if not gated:
            return []   # abstention: nothing both flips the model and passes the gate

        # Stage 2 -- manifold density, falling back to the densest half.
        dense = [i for i in gated if self._dense(Z[i])]
        if not dense:
            scores = [self._lof.score_samples((Z[i] / self.spec.ranges).reshape(1, -1))[0]
                      for i in gated]
            top = np.argsort(scores)[::-1][: max(1, len(gated) // 2)]
            dense = [gated[j] for j in top]

        # Stage 3 -- confidence margin, falling back to plain validity.
        pg = p[dense]
        thresh = tau_target if pg.max() >= tau_target else 0.50 + 1e-9
        return [(lib[i], Z[i]) for i in dense if p[i] >= thresh]

    def tighten(self, xv, bundle, z0, use_causal=True, use_snap=True, tau=None):
        """Greedily drop or shrink interventions that do no work.

        Each proposal must clear ``min(tau(x), p(y'|z0))`` rather than
        ``tau(x)`` alone, so a candidate is held to the margin it already had;
        without that floor, tightening would walk every candidate back toward
        the decision boundary.
        """
        tau_target = self._tau(xv) if tau is None else tau
        p0 = float(self._p_class(z0.reshape(1, -1))[0])
        thresh = min(tau_target, p0)
        cur_b, cur_z = dict(bundle), z0

        while True:
            moves = []
            for f in cur_b:
                rest = {kk: vv for kk, vv in cur_b.items() if kk != f}
                if rest:
                    moves.append(rest)
                if abs(cur_b[f]) >= 2:
                    moves.append({**cur_b, f: cur_b[f] - int(np.sign(cur_b[f]))})
            if not moves:
                return cur_b, cur_z

            Z = self._resolve(xv, moves, use_causal, use_snap)
            p = self._p_class(Z)
            ok = [(self._gower(xv, Z[i]), moves[i], Z[i])
                  for i in np.flatnonzero(p >= thresh)
                  if self._gate_ok(xv, moves[i], Z[i]) and self._dense(Z[i])]
            if not ok:
                return cur_b, cur_z
            ok.sort(key=lambda t: t[0])
            _, cur_b, cur_z = ok[0]

    # -- Pareto selection ---------------------------------------------------

    def _objectives(self, xv, z, copies=()) -> Tuple[float, float, float, float]:
        """All lower-is-better: (gower, l0, 1 - p_target, implausibility).

        The two query-dependent terms are averaged over the query and its
        smoothing copies. Causal violations are excluded because the pool has
        already gated them to zero, so they cannot discriminate.
        """
        qs = [xv] + list(copies)
        g = float(np.mean([self._gower(q, z) for q in qs]))
        l = float(np.mean([self._l0(q, z) for q in qs]))
        return (g, l, 1.0 - self._p_target(z), self._implaus(z))

    @staticmethod
    def _dominates(a, b) -> bool:
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def _front(self, objs) -> List[int]:
        n = len(objs)
        return [i for i in range(n)
                if not any(self._dominates(objs[j], objs[i])
                           for j in range(n) if j != i)]

    @staticmethod
    def _normalize(objs) -> np.ndarray:
        A = np.asarray(objs, dtype=float)
        lo, hi = A.min(axis=0), A.max(axis=0)
        rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        N = (A - lo) / rng
        N[:, hi - lo <= 1e-12] = 0.0
        return N

    def _knee(self, objs) -> int:
        """The front member nearest the ideal point after normalisation."""
        return int(np.argmin(np.linalg.norm(self._normalize(objs), axis=1)))

    def _crowding(self, objs) -> np.ndarray:
        """NSGA-II crowding distance; boundary members get infinity."""
        N = self._normalize(objs)
        n, m = N.shape
        d = np.zeros(n)
        if n <= 2:
            return np.full(n, np.inf)
        for j in range(m):
            order = np.argsort(N[:, j])
            d[order[0]] = d[order[-1]] = np.inf
            span = N[order[-1], j] - N[order[0], j]
            if span <= 1e-12:
                continue
            for r in range(1, n - 1):
                d[order[r]] += (N[order[r + 1], j] - N[order[r - 1], j]) / span
        return d

    def _smooth_copies(self, xv) -> List[np.ndarray]:
        """Deterministic perturbed copies, seeded from the query's own bytes."""
        sp = self.spec
        rng = np.random.default_rng((self.seed << 16) ^ zlib.crc32(xv.tobytes()))
        out = []
        for _ in range(self.j_smooth):
            for _try in range(6):
                f = sp.mutable[int(rng.integers(len(sp.mutable)))]
                jx = sp.index[f]
                lo, hi = sp.limits[f]
                v = float(np.clip(xv[jx] + rng.choice([-1.0, 1.0]), lo, hi))
                if v != xv[jx]:
                    p = xv.copy()
                    p[jx] = v
                    out.append(p)
                    break
        return out

    # -- entry point --------------------------------------------------------

    def explain(
        self,
        xv: np.ndarray,
        *,
        use_causal: bool = True,
        use_snap: bool = True,
        use_tighten: bool = True,
        use_smooth: bool = True,
    ) -> List[np.ndarray]:
        """Return up to ``k`` counterfactuals for ``xv``.

        An empty list is an abstention: no intervention in the library both
        flipped the classifier and satisfied the declared rules. Abstentions
        are scored as failures in the paper's evaluation; recourse that
        violates its own stated guarantees is worse than no recourse.
        """
        xv = np.asarray(xv, dtype=float)
        pool = self.candidate_pool(xv, use_causal, use_snap)
        if not pool:
            return []

        copies = self._smooth_copies(xv) if use_smooth else ()

        if use_tighten:
            seen, extra = set(), []
            for b, z in pool[: self.refine_n]:
                tb, tz = self.tighten(xv, b, z, use_causal, use_snap)
                key = tuple(sorted(tb.items()))
                if key not in seen:
                    seen.add(key)
                    extra.append((tb, tz))
            pool = extra + [c for c in pool
                            if tuple(sorted(c[0].items())) not in seen]

        objs = [self._objectives(xv, z, copies) for _, z in pool]
        l0s = [self._l0(xv, z) for _, z in pool]
        min_l0 = min(l0s)

        # Sparsity is treated lexicographically, not as one objective of four:
        # acting on two attributes is qualitatively easier than acting on three.
        keep = [i for i, l in enumerate(l0s) if l <= min_l0]
        s_objs = [objs[i] for i in keep]
        s_pool = [pool[i] for i in keep]

        front = self._front(s_objs)
        f_objs = [s_objs[i] for i in front]
        f_z = [s_pool[i][1] for i in front]

        first = self._knee(f_objs)
        selected = [f_z[first]]

        crowd = self._crowding(f_objs)
        order = sorted((i for i in range(len(f_z)) if i != first),
                       key=lambda i: -crowd[i])
        for i in order:
            if len(selected) >= self.k:
                break
            if all(self._gower(f_z[i], s) > self.eps for s in selected):
                selected.append(f_z[i])
        return selected
