"""Checks that the generalised rule engine reproduces the notebook's
hard-coded turnover-intention propagate()/causal_violations(), and that the
packaged explainer runs end to end."""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scaffold_cf import FeatureSpec, CausalRule, ScaffoldCF

FEATURES = ["age", "gender", "education", "working_hour", "monthly_salary",
            "satisfaction_with_workload", "satisfied_with_worklife_balance",
            "family_supports", "mentally_well_and_do_not_have_anxiety"]
FIDX = {f: j for j, f in enumerate(FEATURES)}
LIMS = {f: (1.0, 5.0) for f in FEATURES}
WELLNESS = "mentally_well_and_do_not_have_anxiety"
ACTIVE_RULES = [("working_hour", "satisfied_with_worklife_balance"),
                ("satisfaction_with_workload", WELLNESS),
                ("satisfied_with_worklife_balance", WELLNESS)]

def nb_propagate(xv, zv):                      # verbatim from notebook cell 46
    z = zv.copy()
    j_hr, j_wlb = FIDX["working_hour"], FIDX["satisfied_with_worklife_balance"]
    j_wl, j_mw = FIDX["satisfaction_with_workload"], FIDX[WELLNESS]
    for _ in range(3):
        if ("working_hour", "satisfied_with_worklife_balance") in ACTIVE_RULES and z[j_hr] < xv[j_hr]:
            z[j_wlb] = max(z[j_wlb], min(LIMS["satisfied_with_worklife_balance"][1], xv[j_wlb] + 1))
        if ("satisfaction_with_workload", WELLNESS) in ACTIVE_RULES and z[j_wl] - xv[j_wl] >= 2:
            z[j_mw] = max(z[j_mw], min(LIMS[WELLNESS][1], xv[j_mw] + 1))
        if ("satisfied_with_worklife_balance", WELLNESS) in ACTIVE_RULES and z[j_wlb] - xv[j_wlb] >= 2:
            z[j_mw] = max(z[j_mw], min(LIMS[WELLNESS][1], xv[j_mw] + 1))
    return z

def nb_violations(xv, zv):                     # verbatim from notebook cell 46
    v = 0
    j_hr, j_wlb = FIDX["working_hour"], FIDX["satisfied_with_worklife_balance"]
    j_wl, j_mw = FIDX["satisfaction_with_workload"], FIDX[WELLNESS]
    if zv[j_hr] < xv[j_hr] and zv[j_wlb] < min(LIMS["satisfied_with_worklife_balance"][1], xv[j_wlb] + 1):
        v += 1
    if zv[j_wl] - xv[j_wl] >= 2 and zv[j_mw] < min(LIMS[WELLNESS][1], xv[j_mw] + 1):
        v += 1
    if zv[j_wlb] - xv[j_wlb] >= 2 and zv[j_mw] < min(LIMS[WELLNESS][1], xv[j_mw] + 1):
        v += 1
    return v

SPEC = FeatureSpec(FEATURES, LIMS, immutable=["age", "gender", "education"],
                   downstream_only=["family_supports", WELLNESS])
RULES = [CausalRule("working_hour", "satisfied_with_worklife_balance", -1, 1),
         CausalRule("satisfaction_with_workload", WELLNESS, +1, 2),
         CausalRule("satisfied_with_worklife_balance", WELLNESS, +1, 2)]

def main():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.integers(1, 6, size=(400, len(FEATURES))).astype(float),
                     columns=FEATURES)
    y = ((X["working_hour"] > 3).astype(int)
         | (X["satisfaction_with_workload"] < 2).astype(int)).values
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=60, random_state=0).fit(X.values, y)
    ex = ScaffoldCF(model, X, SPEC, RULES, desired_class=0, seed=42).fit()

    bad_p = bad_v = 0
    for _ in range(20000):
        xv = rng.integers(1, 6, len(FEATURES)).astype(float)
        zv = rng.integers(1, 6, len(FEATURES)).astype(float)
        if not np.array_equal(ex.propagate(xv, zv), nb_propagate(xv, zv)): bad_p += 1
        if ex.causal_violations(xv, zv) != nb_violations(xv, zv): bad_v += 1
    print(f"propagate mismatches        : {bad_p} / 20000")
    print(f"causal_violations mismatches: {bad_v} / 20000")

    print(f"library size                : {len(ex._library)}")
    n_abstain = 0
    for i in range(25):
        cfs = ex.explain(X.values[i])
        if not cfs:
            n_abstain += 1
            continue
        for z in cfs:
            assert ex.causal_violations(X.values[i], z) == 0, "C2 violated"
            assert model.predict(z.reshape(1, -1))[0] == 0, "not valid"
            for f in SPEC.immutable:
                assert z[FIDX[f]] == X.values[i][FIDX[f]], f"immutable {f} changed"
    print(f"25 queries                  : {n_abstain} abstentions, "
          f"all returned CFs valid, causally admissible, immutables untouched")
    assert bad_p == 0 and bad_v == 0
    print("\nOK")

main()
