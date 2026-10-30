"""Are aggregate ranking metrics blind to the loss of the sign-discordant class?

Reports, per replicate, (i) aggregate precision at k, (ii) detection of the
sign-discordant class relative to the concordant class, under pooling and under
state conditioning. Computed per replicate, then summarised.
"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

def run(seed, G=2000, n_total=40, pi1=0.6, sigma=1.0, b=1.0, r=1.0,
        f_conc=0.08, f_spec=0.06, f_disc=0.06, k=50):
    rng = np.random.default_rng(seed)
    cls = rng.choice(4, size=G, p=[1 - f_conc - f_spec - f_disc, f_conc, f_spec, f_disc])
    beta = np.zeros((G, 2))
    beta[cls == 1] = [b, b]; beta[cls == 2] = [b, 0.0]; beta[cls == 3] = [b, -r * b]
    n1 = int(round(n_total * pi1)); n2 = n_total - n1
    y1 = beta[:, [0]] + rng.normal(0, sigma, (G, n1))
    y2 = beta[:, [1]] + rng.normal(0, sigma, (G, n2))
    e1 = y1.mean(1); ep = np.concatenate([y1, y2], axis=1).mean(1)
    out = {}
    for name, score in (('state_conditioned', e1), ('pooled', ep)):
        rank = np.empty(G, int); rank[np.argsort(-score)] = np.arange(1, G + 1)
        top = rank <= k
        # aggregate precision: is the selected gene synthetic lethal in state 1?
        out[f'{name}_precision_at_k'] = float((beta[top, 0] > 0).mean())
        det_d = (rank[cls == 3] <= k).mean()
        det_c = (rank[cls == 1] <= k).mean()
        out[f'{name}_detection_discordant'] = float(det_d)
        out[f'{name}_detection_concordant'] = float(det_c)
        out[f'{name}_discordant_to_concordant_ratio'] = float(det_d / det_c) if det_c else float('nan')
    return out

N = 500
rows = [run(s) for s in range(N)]
summary = {}
for key in rows[0]:
    a = np.array([x[key] for x in rows], float)
    summary[key] = {'mean': round(float(np.nanmean(a)), 4),
                    'sd': round(float(np.nanstd(a, ddof=1)), 4),
                    'ci95': [round(float(np.nanpercentile(a, 2.5)), 4),
                             round(float(np.nanpercentile(a, 97.5)), 4)]}
summary['n_replicates'] = N
(ROOT / 'results' / 'metric_blindness.json').write_text(json.dumps(summary, indent=2))
for k_, v in summary.items():
    if k_ == 'n_replicates':
        continue
    print(f"{k_:52s} {v['mean']:.4f} ± {v['sd']:.4f}  [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
