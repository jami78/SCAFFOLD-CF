# SCAFFOLD-CF

Causally-constrained counterfactual recourse. **S**tructurally **C**onstrained
**A**ctionable **F**iltering **F**or **O**n-manifold **L**ow-cost **D**ecisions.

Most counterfactual methods treat features as independently mutable and
produce recourse that conflicts with the causal relations declared for a
domain. SCAFFOLD-CF enforces causal admissibility as a **hard filter** rather
than as a penalty term: a candidate is returned only if it flips the
classifier, touches only actionable features, and leaves no declared rule
unsatisfied. When nothing survives, the method abstains.

> The causal rule set is a **declared input, not a validated causal model**.
> The conditions below are properties of a candidate relative to that rule
> set; they are not claims about what would be observed after intervening.

## Install

```bash
pip install -r requirements.txt
```

## Use

```python
from scaffold_cf import FeatureSpec, CausalRule, ScaffoldCF

spec = FeatureSpec(
    features=list(X_train.columns),
    limits={f: (X_train[f].min(), X_train[f].max()) for f in X_train},
    immutable=["age", "gender", "education"],          # never intervened on
    downstream_only=["family_supports", "mentally_well"],  # consequence only
)

rules = [
    # when working_hour falls by >= 1, work_life_balance may not stay put
    CausalRule("working_hour", "work_life_balance", direction=-1, threshold=1),
    # when workload satisfaction rises by >= 2, wellbeing may not stay put
    CausalRule("workload_satisfaction", "mentally_well", direction=+1, threshold=2),
]

explainer = ScaffoldCF(model, X_train, spec, rules, desired_class=0).fit()
cfs = explainer.explain(x)      # list of arrays; [] is an abstention
```

`model` is any fitted classifier with `predict`, `predict_proba` and
`classes_`. Nothing needs to be differentiable.

## How it works

1. **Library** — mined transitions from unfavourable to favourable training
   rows, plus enumerated single, pairwise and triple edits, capped at three
   features and ordered sparsest-first.
2. **Resolution** — `Phi(x,b) = Snap(Propagate(Clip(x+b)))`. Propagation lifts
   each rule's effect to its floor, to a fixed point, so chained rules fire.
3. **Gates** — validity, actionability, C1 (every change is intervened on or
   rule-entailed), C2 (no rule left unsatisfied), LOF density, and a
   query-scaled confidence margin.
4. **Tightening** — greedily drops interventions that do no work, held to the
   margin the candidate already had.
5. **Selection** — lexicographic sparsity stratification, then a Pareto front
   over (proximity, sparsity, target proximity, implausibility), then the knee
   point and crowding distance with a diversity threshold.

## Abstention

`explain()` returns `[]` when no intervention both flips the classifier and
satisfies the rules. This costs coverage, and the cost is deliberate.

The original notebooks also relaxed the causal conditions when the gated pool
came up empty. That fallback is **off by default** here, because auditing
every query-seed across five seeds showed it was reached on 20 of 196
turnover-intention queries and on every occasion the relaxed pool was empty
too — so the method abstained rather than returning a relaxed candidate, and
no published result came through that path. Leaving it off makes the causal
guarantee unconditional. Pass `relax_causal_on_empty=True` to restore it.

## Tests

```bash
python tests/test_equivalence.py
```

Checks the generalised rule engine against the hard-coded version from the
benchmark notebooks over 20,000 random pairs, and runs the explainer end to
end asserting validity, causal admissibility and immutability.

