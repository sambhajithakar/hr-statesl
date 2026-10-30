"""Extended estimation simulation.

The baseline run in 02 showed that under pooling the dominant error is not the
promotion of anti-targets but the silent loss of sign-discordant genes, whose
pooled estimate is attenuated out of the candidate list. This script quantifies
that loss directly (detection and rank of the discordant class) and measures the
transfer error incurred when one pooled list is applied to a single HR state,
across the mixture of concordant and state-specific synthetic lethality.
"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / 'results'

def simulate(rng, G, f_conc, f_spec, f_disc, b, r):
    f_neu = 1.0 - f_conc - f_spec - f_disc
    assert f_neu > 0
    cls = rng.choice(4, size=G, p=[f_neu, f_conc, f_spec, f_disc])
    beta = np.zeros((G, 2))
    beta[cls == 1] = [b, b]
    beta[cls == 2] = [b, 0.0]
    beta[cls == 3] = [b, -r * b]
    return cls, beta

def run(seed, G=2000, n_total=40, pi1=0.6, sigma=1.0, b=1.0, r=1.0,
        f_conc=0.08, f_spec=0.06, f_disc=0.06, k=50):
    rng = np.random.default_rng(seed)
    cls, beta = simulate(rng, G, f_conc, f_spec, f_disc, b, r)
    n1 = int(round(n_total * pi1)); n2 = n_total - n1
    y1 = beta[:, [0]] + rng.normal(0, sigma, (G, n1))
    y2 = beta[:, [1]] + rng.normal(0, sigma, (G, n2))
    e1, e2 = y1.mean(1), y2.mean(1)
    ep = np.concatenate([y1, y2], axis=1).mean(1)
    d = cls == 3
    o = {}

    # --- detection of the sign-discordant class in the state where it is lethal
    for name, score in (('state_conditioned', e1), ('pooled', ep)):
        rank = np.empty(G, int); rank[np.argsort(-score)] = np.arange(1, G + 1)
        o[f'{name}_discordant_detected_at_k'] = float((rank[d] <= k).mean())
        o[f'{name}_discordant_median_rank'] = float(np.median(rank[d]))
        o[f'{name}_concordant_detected_at_k'] = float((rank[cls == 1] <= k).mean())

    # --- transfer error: ONE pooled list, applied to state 2
    top_p = np.argsort(-ep)[:k]
    top_c = np.argsort(-e2)[:k]
    o['pooled_transfer_not_SL_in_state2'] = float((beta[top_p, 1] <= 0).mean())
    o['pooled_transfer_anti_SL_in_state2'] = float((beta[top_p, 1] < 0).mean())
    o['conditioned_not_SL_in_state2'] = float((beta[top_c, 1] <= 0).mean())
    o['conditioned_anti_SL_in_state2'] = float((beta[top_c, 1] < 0).mean())

    # --- attenuation measured versus the closed-form prediction of section A
    o['measured_attenuation_discordant'] = float(np.abs(ep[d]).mean() /
                                                 np.abs(e1[d]).mean())
    o['predicted_attenuation_discordant'] = float(abs(pi1 - r * (1 - pi1)) / max(1.0, r))
    return o

def agg(ds):
    out = {}
    for key in ds[0]:
        a = np.array([x[key] for x in ds], float)
        out[key] = {'mean': round(float(a.mean()), 4), 'sd': round(float(a.std(ddof=1)), 4),
                    'ci95': [round(float(np.percentile(a, 2.5)), 4),
                             round(float(np.percentile(a, 97.5)), 4)]}
    return out

N = 200
OUT = {'n_replicates': N,
       'baseline': agg([run(s) for s in range(N)]),
       'sweep_reversal_ratio': {f'r={r}': agg([run(s, r=r) for s in range(N)])
                                for r in (0.25, 0.5, 1.0, 1.5, 2.0)},
       'sweep_state_prevalence': {f'pi1={p}': agg([run(s, pi1=p) for s in range(N)])
                                  for p in (0.5, 0.6, 0.7)},
       'sweep_concordant_fraction': {f'f_conc={f}': agg([run(s, f_conc=f) for s in range(N)])
                                     for f in (0.02, 0.04, 0.08, 0.12)},
       'sweep_screen_noise': {f'sigma={s_}': agg([run(s, sigma=s_) for s in range(N)])
                              for s_ in (0.5, 1.0, 2.0)},
       'sweep_cell_lines': {f'n_total={n}': agg([run(s, n_total=n) for s in range(N)])
                            for n in (20, 40, 80, 160)}}
(RES / 'estimation_extended.json').write_text(json.dumps(OUT, indent=2))

b = OUT['baseline']
def g(k): return f"{b[k]['mean']:.3f} ± {b[k]['sd']:.3f}"
print('--- baseline, k=50, discordant class ---')
print('detected at k   conditioned', g('state_conditioned_discordant_detected_at_k'),
      '| pooled', g('pooled_discordant_detected_at_k'))
print('median rank     conditioned', g('state_conditioned_discordant_median_rank'),
      '| pooled', g('pooled_discordant_median_rank'))
print('concordant@k    conditioned', g('state_conditioned_concordant_detected_at_k'),
      '| pooled', g('pooled_concordant_detected_at_k'))
print('attenuation measured', g('measured_attenuation_discordant'),
      'predicted', g('predicted_attenuation_discordant'))
print('transfer: pooled list not-SL in state2', g('pooled_transfer_not_SL_in_state2'),
      '| conditioned', g('conditioned_not_SL_in_state2'))
print('\n--- transfer error vs concordant fraction ---')
for k, v in OUT['sweep_concordant_fraction'].items():
    print(f"  {k:12s} pooled not-SL {v['pooled_transfer_not_SL_in_state2']['mean']:.3f}"
          f"  anti-SL {v['pooled_transfer_anti_SL_in_state2']['mean']:.3f}"
          f" | conditioned not-SL {v['conditioned_not_SL_in_state2']['mean']:.3f}")
print('\n--- discordant detection vs r ---')
for k, v in OUT['sweep_reversal_ratio'].items():
    print(f"  {k:8s} pooled {v['pooled_discordant_detected_at_k']['mean']:.3f}"
          f" (rank {v['pooled_discordant_median_rank']['mean']:.0f})"
          f" | conditioned {v['state_conditioned_discordant_detected_at_k']['mean']:.3f}"
          f" (rank {v['state_conditioned_discordant_median_rank']['mean']:.0f})")
print('\n--- discordant detection vs cell lines ---')
for k, v in OUT['sweep_cell_lines'].items():
    print(f"  {k:14s} pooled {v['pooled_discordant_detected_at_k']['mean']:.3f}"
          f" | conditioned {v['state_conditioned_discordant_detected_at_k']['mean']:.3f}")
