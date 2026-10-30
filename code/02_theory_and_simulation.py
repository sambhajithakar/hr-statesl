"""Theory, simulation and design calculations for state-conditioned synthetic
lethality prioritization.

Sections
  A  closed-form attenuation and identifiability under state pooling
  B  estimation simulation: pooled versus state-stratified candidate lists
  C  learning simulation: identical data, with and without the state covariate
  D  design calculation: replicates needed to confirm a sign reversal

Every number reported in the manuscript Results comes from this file. Sections
B and C are SIMULATIONS under the stated generative model; they are not
measurements of real cell lines and are labelled as such throughout.
"""
import json, pathlib
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
RES.mkdir(exist_ok=True)
OUT = {}

# ---------------------------------------------------------------- A: theory
# Under pooling, the estimand for gene g is sum_s pi_s * beta_{g,s}.
# With two states and beta = (b, -r*b), the pooled estimand is b*(pi1 - r*pi2).
# Attenuation A = |pooled| / max_s |beta_s|.  A = 0 exactly when r = pi1/pi2.
def attenuation(pi1, r):
    pi2 = 1.0 - pi1
    pooled = abs(pi1 - r * pi2)
    return pooled / max(1.0, r)

grid_pi = [0.50, 0.55, 0.60, 0.65, 0.70]
grid_r = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
A_table = {f'pi1={p:.2f}': {f'r={r:.2f}': round(attenuation(p, r), 4)
                            for r in grid_r} for p in grid_pi}
# Sample-size inflation for pooled detection relative to the stratified test of
# the stronger arm: variance of a mean scales as 1/n, so n scales as 1/A^2.
infl = {f'pi1={p:.2f}': {f'r={r:.2f}': (None if attenuation(p, r) == 0
                                        else round(1.0 / attenuation(p, r) ** 2, 2))
                         for r in grid_r} for p in grid_pi}
OUT['A_theory'] = {
    'statement': ('pooled estimand equals the prevalence-weighted mean of the '
                  'per-state effects; it vanishes at r = pi1/pi2 regardless of '
                  'how large the per-state effects are'),
    'attenuation_table': A_table,
    'pooled_sample_size_inflation_factor': infl,
    'cancellation_ratio_r_star': {f'pi1={p:.2f}': round(p / (1 - p), 4) for p in grid_pi},
}

# ----------------------------------------------- B: estimation simulation
CLASSES = ('neutral', 'concordant_SL', 'state_specific_SL', 'sign_discordant')

def simulate_effects(rng, G, f_disc, b, r):
    """Assign each gene a true per-state effect vector (state 0, state 1)."""
    f_conc, f_spec = 0.08, 0.06
    f_neu = 1.0 - f_conc - f_spec - f_disc
    assert f_neu > 0, 'class fractions exceed 1'
    cls = rng.choice(len(CLASSES), size=G, p=[f_neu, f_conc, f_spec, f_disc])
    beta = np.zeros((G, 2))
    beta[cls == 1] = [b, b]
    beta[cls == 2] = [b, 0.0]
    beta[cls == 3] = [b, -r * b]
    return cls, beta

def run_estimation(seed, G=2000, n_total=40, pi1=0.6, sigma=1.0,
                   f_disc=0.06, b=1.0, r=1.0, k=50):
    rng = np.random.default_rng(seed)
    cls, beta = simulate_effects(rng, G, f_disc, b, r)
    n1 = int(round(n_total * pi1)); n2 = n_total - n1
    y1 = beta[:, [0]] + rng.normal(0, sigma, (G, n1))
    y2 = beta[:, [1]] + rng.normal(0, sigma, (G, n2))
    est_s = np.column_stack([y1.mean(1), y2.mean(1)])            # stratified
    est_p = np.concatenate([y1, y2], axis=1).mean(1)             # pooled
    out = {}
    for name, score in (('state_conditioned', est_s[:, 1]), ('pooled', est_p)):
        top = np.argsort(-score)[:k]
        true = beta[top, 1]                                       # truth in state 2
        out[name] = {
            'precision_at_k': float((true > 0).mean()),
            'anti_target_rate_at_k': float((true < 0).mean()),
            'recall_at_k_state2_SL': float((true > 0).sum() / max(1, (beta[:, 1] > 0).sum())),
        }
    # Recovery of the discordant class specifically
    d = cls == 3
    out['discordant_sign_recovered_state_conditioned'] = float(
        ((est_s[d, 0] > 0) & (est_s[d, 1] < 0)).mean())
    out['discordant_sign_recovered_pooled'] = 0.0  # a single score cannot encode two signs
    out['mean_abs_pooled_estimate_discordant'] = float(np.abs(est_p[d]).mean())
    out['mean_abs_stratified_estimate_discordant'] = float(np.abs(est_s[d]).mean())
    return out

def aggregate(dicts):
    keys = dicts[0].keys()
    agg = {}
    for key in keys:
        v = dicts[0][key]
        if isinstance(v, dict):
            agg[key] = aggregate([d[key] for d in dicts])
        else:
            a = np.array([d[key] for d in dicts], float)
            agg[key] = {'mean': round(float(a.mean()), 4),
                        'sd': round(float(a.std(ddof=1)), 4),
                        'ci95': [round(float(np.percentile(a, 2.5)), 4),
                                 round(float(np.percentile(a, 97.5)), 4)]}
    return agg

N_REP = 200
base = aggregate([run_estimation(s) for s in range(N_REP)])
OUT['B_estimation'] = {
    'n_replicates': N_REP,
    'generative_parameters': {'G': 2000, 'n_cell_lines_total': 40, 'pi1': 0.6,
                              'sigma': 1.0, 'effect_size_b': 1.0,
                              'reversal_ratio_r': 1.0, 'f_sign_discordant': 0.06,
                              'f_concordant': 0.08, 'f_state_specific': 0.06, 'k': 50},
    'baseline': base,
}
# sweeps
sweeps = {}
for f_disc in (0.02, 0.06, 0.10, 0.15):
    sweeps[f'f_disc={f_disc}'] = aggregate(
        [run_estimation(s, f_disc=f_disc) for s in range(N_REP)])
for r in (0.5, 1.0, 1.5):
    sweeps[f'r={r}'] = aggregate([run_estimation(s, r=r) for s in range(N_REP)])
for pi1 in (0.5, 0.6, 0.7):
    sweeps[f'pi1={pi1}'] = aggregate([run_estimation(s, pi1=pi1) for s in range(N_REP)])
for sigma in (0.5, 1.0, 2.0):
    sweeps[f'sigma={sigma}'] = aggregate([run_estimation(s, sigma=sigma) for s in range(N_REP)])
OUT['B_sweeps'] = sweeps

# ------------------------------------------------- C: learning simulation
def run_learning(seed, G=4000, p=24, f_disc=0.06, noise=0.6, k=50):
    """Identical training observations for both models; one lacks the state
    covariate. Labels are per-(gene,state) synthetic-lethal indicators."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(G, p))
    w1, w2 = rng.normal(size=p), rng.normal(size=p)
    cls, _ = simulate_effects(rng, G, f_disc, 1.0, 1.0)
    # latent per-state effect driven by features; discordant genes get a flipped
    # second-state map so the sign reversal is itself feature-predictable
    lat1 = X @ w1
    lat2 = np.where(cls == 3, -(X @ w1), X @ w2)
    thr1, thr2 = np.quantile(lat1, 0.88), np.quantile(lat2, 0.88)
    y1 = (lat1 + rng.normal(0, noise, G) > thr1).astype(int)
    y2 = (lat2 + rng.normal(0, noise, G) > thr2).astype(int)
    idx = rng.permutation(G); tr, te = idx[:G // 2], idx[G // 2:]

    Xs = np.vstack([np.column_stack([X, np.zeros(G)]),
                    np.column_stack([X, np.ones(G)])])
    ys = np.concatenate([y1, y2])
    rows_tr = np.concatenate([tr, tr + G]); rows_te = np.concatenate([te, te + G])

    res = {}
    for name, cols in (('state_conditioned', slice(None)), ('state_agnostic', slice(0, p))):
        m = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        m.fit(Xs[np.ix_(rows_tr)][:, cols], ys[rows_tr])
        sc = m.predict_proba(Xs[np.ix_(rows_te)][:, cols])[:, 1]
        sc1, sc2 = sc[:len(te)], sc[len(te):]
        res[name] = {
            'auprc_state1': float(average_precision_score(y1[te], sc1)),
            'auprc_state2': float(average_precision_score(y2[te], sc2)),
            'auroc_state2': float(roc_auc_score(y2[te], sc2)),
        }
        # anti-target rate: genes ranked in the top k for state 2 that are in
        # truth synthetic lethal in state 1 and NOT synthetic lethal in state 2
        top = te[np.argsort(-sc2)[:k]]
        res[name]['top_k_state2_wrong_state_hits'] = float(
            ((y1[top] == 1) & (y2[top] == 0)).mean())
        dte = te[cls[te] == 3]
        if len(dte):
            s1 = sc1[np.isin(te, dte)]; s2 = sc2[np.isin(te, dte)]
            res[name]['discordant_score_gap'] = float(np.mean(s1 - s2))
    return res

N_REP_C = 30
OUT['C_learning'] = {
    'n_replicates': N_REP_C,
    'generative_parameters': {'G': 4000, 'n_features': 24, 'f_sign_discordant': 0.06,
                              'label_noise_sd': 0.6, 'positive_rate': 0.12,
                              'model': 'HistGradientBoostingClassifier(max_iter=200)',
                              'split': 'random 50/50 gene-disjoint', 'k': 50},
    'results': aggregate([run_learning(s) for s in range(N_REP_C)]),
}
OUT['C_learning_f_disc_sweep'] = {
    f'f_disc={f}': aggregate([run_learning(s, f_disc=f) for s in range(N_REP_C)])
    for f in (0.02, 0.10, 0.15)
}

# ------------------------------------------------------ D: design calculation
def n_for_power(d, power=0.80, alpha=0.05, n_tests=2, nmax=400):
    """Replicates per arm for a one-sample t-test of effect d = |beta|/sigma,
    one-sided, Bonferroni-corrected across n_tests arms."""
    a = alpha / n_tests
    for n in range(3, nmax + 1):
        crit = stats.t.ppf(1 - a, n - 1)
        pw = stats.nct.sf(crit, n - 1, d * np.sqrt(n))
        if pw >= power:
            return n, round(float(pw), 4)
    return None, None

design = {}
for d in (0.5, 0.8, 1.0, 1.5, 2.0):
    n_arm, pw = n_for_power(d)
    joint = None if n_arm is None else round(pw ** 2, 4)
    design[f'standardised_effect={d}'] = {
        'n_replicates_per_arm': n_arm,
        'power_per_arm': pw,
        'joint_power_both_signs': joint,
    }
# n needed so that the JOINT (both-arm) power reaches 0.80
design_joint = {}
for d in (0.5, 0.8, 1.0, 1.5, 2.0):
    for n in range(3, 401):
        crit = stats.t.ppf(1 - 0.025, n - 1)
        pw = stats.nct.sf(crit, n - 1, d * np.sqrt(n))
        if pw ** 2 >= 0.80:
            design_joint[f'standardised_effect={d}'] = {
                'n_replicates_per_arm': n, 'joint_power': round(float(pw ** 2), 4)}
            break
OUT['D_design'] = {'per_arm_80pct': design, 'joint_80pct': design_joint,
                   'test': 'one-sided one-sample t-test per HR state, '
                           'Bonferroni alpha/2, sign reversal declared only if '
                           'both arms reject in opposite directions'}

(RES / 'theory_and_simulation.json').write_text(json.dumps(OUT, indent=2))
print(json.dumps({'A_theory': OUT['A_theory'], 'B_baseline': OUT['B_estimation']['baseline'],
                  'C_results': OUT['C_learning']['results'], 'D_joint': OUT['D_design']['joint_80pct']},
                 indent=2))
