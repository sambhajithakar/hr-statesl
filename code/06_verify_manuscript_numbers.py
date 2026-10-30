"""Check every numeric claim in the manuscript against the stored result files."""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
R = lambda n: json.loads((ROOT / 'results' / n).read_text())
T, E, B, S = R('theory_and_simulation.json'), R('estimation_extended.json'), R('metric_blindness.json'), R('resource_stats.json')
fails = []
def chk(label, claimed, actual, tol=5e-4):
    ok = abs(claimed - actual) <= tol
    print(f"{'OK ' if ok else 'FAIL'}  {label:58s} manuscript={claimed}  stored={actual}")
    if not ok: fails.append(label)

b = E['baseline']
chk('attenuation measured mean', 0.215, b['measured_attenuation_discordant']['mean'])
chk('attenuation measured sd', 0.011, b['measured_attenuation_discordant']['sd'])
chk('attenuation predicted', 0.200, b['predicted_attenuation_discordant']['mean'])
chk('discordant median rank conditioned', 199, round(b['state_conditioned_discordant_median_rank']['mean']), 0.5)
chk('discordant median rank cond sd', 17, round(b['state_conditioned_discordant_median_rank']['sd']), 0.5)
chk('discordant median rank pooled', 505, round(b['pooled_discordant_median_rank']['mean']), 0.5)
chk('discordant median rank pooled sd', 38, round(b['pooled_discordant_median_rank']['sd']), 0.5)
chk('transfer pooled not-SL', 0.004, b['pooled_transfer_not_SL_in_state2']['mean'])
chk('transfer pooled sd', 0.009, b['pooled_transfer_not_SL_in_state2']['sd'])
chk('transfer conditioned not-SL', 0.000, b['conditioned_not_SL_in_state2']['mean'])
chk('transfer conditioned sd', 0.002, b['conditioned_not_SL_in_state2']['sd'])
for r, v in zip((0.25, 0.5, 1.0, 1.5, 2.0), (310, 337, 505, 1135, 1772)):
    chk(f'pooled median rank r={r}', v,
        round(E['sweep_reversal_ratio'][f'r={r}']['pooled_discordant_median_rank']['mean']), 0.5)
for n in (20, 40, 80, 160):
    chk(f'pooled detection n={n}', 0.000,
        E['sweep_cell_lines'][f'n_total={n}']['pooled_discordant_detected_at_k']['mean'])
chk('transfer f_conc=0.02 pooled', 0.283, E['sweep_concordant_fraction']['f_conc=0.02']['pooled_transfer_not_SL_in_state2']['mean'])
chk('transfer f_conc=0.02 conditioned', 0.252, E['sweep_concordant_fraction']['f_conc=0.02']['conditioned_not_SL_in_state2']['mean'])

chk('precision@50 pooled', 1.000, B['pooled_precision_at_k']['mean'])
chk('precision@50 conditioned', 1.000, B['state_conditioned_precision_at_k']['mean'])
chk('ratio pooled', 0.000, B['pooled_discordant_to_concordant_ratio']['mean'])
chk('ratio conditioned', 1.072, B['state_conditioned_discordant_to_concordant_ratio']['mean'])
chk('ratio conditioned sd', 0.359, B['state_conditioned_discordant_to_concordant_ratio']['sd'])
chk('ratio CI lo', 0.516, B['state_conditioned_discordant_to_concordant_ratio']['ci95'][0])
chk('ratio CI hi', 1.843, B['state_conditioned_discordant_to_concordant_ratio']['ci95'][1])
chk('detection concordant pooled', 0.313, B['pooled_detection_concordant']['mean'], 1e-3)  # stored 0.3125, rounds to 0.313
chk('detection concordant pooled sd', 0.023, B['pooled_detection_concordant']['sd'])
chk('detection concordant conditioned', 0.124, B['state_conditioned_detection_concordant']['mean'])
chk('detection concordant cond sd', 0.021, B['state_conditioned_detection_concordant']['sd'])
chk('detection discordant conditioned', 0.126, B['state_conditioned_detection_discordant']['mean'])

L = T['C_learning']['results']
for nm, key, c_m, c_s in (('cond auprc1', ('state_conditioned', 'auprc_state1'), 0.791, 0.029),
                          ('agn auprc1', ('state_agnostic', 'auprc_state1'), 0.451, 0.076),
                          ('cond auprc2', ('state_conditioned', 'auprc_state2'), 0.715, 0.040),
                          ('agn auprc2', ('state_agnostic', 'auprc_state2'), 0.406, 0.069),
                          ('cond auroc2', ('state_conditioned', 'auroc_state2'), 0.923, 0.022),
                          ('agn auroc2', ('state_agnostic', 'auroc_state2'), 0.806, 0.033),
                          ('cond wrong', ('state_conditioned', 'top_k_state2_wrong_state_hits'), 0.011, 0.015),
                          ('agn wrong', ('state_agnostic', 'top_k_state2_wrong_state_hits'), 0.344, 0.103)):
    chk(nm + ' mean', c_m, L[key[0]][key[1]]['mean'])
    chk(nm + ' sd', c_s, L[key[0]][key[1]]['sd'])

W = T['C_learning_f_disc_sweep']
chk('f_disc 0.02 agn wrong', 0.326, W['f_disc=0.02']['state_agnostic']['top_k_state2_wrong_state_hits']['mean'])
chk('f_disc 0.15 agn wrong', 0.403, W['f_disc=0.15']['state_agnostic']['top_k_state2_wrong_state_hits']['mean'])
chk('f_disc 0.02 agn auprc2', 0.439, W['f_disc=0.02']['state_agnostic']['auprc_state2']['mean'])
chk('f_disc 0.15 agn auprc2', 0.335, W['f_disc=0.15']['state_agnostic']['auprc_state2']['mean'])
chk('f_disc 0.02 cond auprc2', 0.766, W['f_disc=0.02']['state_conditioned']['auprc_state2']['mean'])
chk('f_disc 0.15 cond auprc2', 0.602, W['f_disc=0.15']['state_conditioned']['auprc_state2']['mean'])
chk('f_disc 0.02 cond wrong', 0.008, W['f_disc=0.02']['state_conditioned']['top_k_state2_wrong_state_hits']['mean'])
chk('f_disc 0.15 cond wrong', 0.026, W['f_disc=0.15']['state_conditioned']['top_k_state2_wrong_state_hits']['mean'])

D = T['D_design']['joint_80pct']
for d, n, p in ((0.5, 44, 0.810), (0.8, 19, 0.827), (1.0, 13, 0.829), (1.5, 7, 0.824), (2.0, 5, 0.826)):
    chk(f'design n d={d}', n, D[f'standardised_effect={d}']['n_replicates_per_arm'], 0)
    chk(f'design power d={d}', p, D[f'standardised_effect={d}']['joint_power'])

A = T['A_theory']
chk('inflation pi=0.60 r=1.00', 25.0, A['pooled_sample_size_inflation_factor']['pi1=0.60']['r=1.00'], 0.05)
chk('inflation pi=0.55 r=1.50', 144.0, A['pooled_sample_size_inflation_factor']['pi1=0.55']['r=1.50'], 0.5)
for p, v in (('0.50', 1.0), ('0.55', 1.2222), ('0.60', 1.5), ('0.65', 1.8571), ('0.70', 2.3333)):
    chk(f'r* pi={p}', round(v, 2), round(A['cancellation_ratio_r_star'][f'pi1={p}'], 2), 0.005)
# Table 6
tab = {'0.50': [0.375, 0.250, 0.125, 0.000, 0.167, 0.250], '0.55': [0.438, 0.325, 0.213, 0.100, 0.083, 0.175],
       '0.60': [0.500, 0.400, 0.300, 0.200, 0.000, 0.100], '0.65': [0.563, 0.475, 0.388, 0.300, 0.083, 0.025],
       '0.70': [0.625, 0.550, 0.475, 0.400, 0.167, 0.050]}
for p, vals in tab.items():
    for rr, v in zip(('0.25', '0.50', '0.75', '1.00', '1.50', '2.00'), vals):
        chk(f'Table6 pi={p} r={rr}', v, A['attenuation_table'][f'pi1={p}'][f'r={rr}'], 0.0006)

for lbl, c, a in (('edges', 26, S['n_edges']), ('genes', 18, S['n_distinct_gene_a']),
                  ('sources', 18, S['n_distinct_sources']), ('preprints', 1, S['n_preprint_sourced']),
                  ('non-breast flagged', 2, S['n_edges_flagged_non_breast_origin']),
                  ('E1', 5, S['by_evidence_class']['E1-interventional-isogenic']),
                  ('E2', 3, S['by_evidence_class']['E2-interventional-screen']),
                  ('E3', 4, S['by_evidence_class']['E3-pharmacological-rescue']),
                  ('E4', 4, S['by_evidence_class']['E4-clinical-observational']),
                  ('E6', 10, S['by_evidence_class']['E6-textual-only']),
                  ('high', 8, S['by_causal_support']['high']), ('medium', 8, S['by_causal_support']['medium']),
                  ('low', 10, S['by_causal_support']['low']),
                  ('interventional', 8, S['n_interventional']), ('target-like', 17, S['n_target_like_edges']),
                  ('target-like low', 7, S['target_like_by_causal_support']['low']),
                  ('anti-SL genes', 3, S['n_anti_sl_genes']), ('discordant genes', 1, S['n_sign_discordant_genes']),
                  ('ctDNA detectable', 14, S['by_ctdna_detectability']['detectable']),
                  ('ctDNA often-missed', 3, S['by_ctdna_detectability']['often-missed']),
                  ('ctDNA signature', 4, S['by_ctdna_detectability']['signature-inferred']),
                  ('ctDNA n/a', 5, S['by_ctdna_detectability']['not-applicable'])):
    chk('resource ' + lbl, c, a, 0)
chk('frac interventional %', 30.8, round(S['frac_interventional'] * 100, 1), 0.05)
chk('frac target-like low %', 41.2, round(S['frac_target_like_low_support'] * 100, 1), 0.05)
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILURES: ' + ', '.join(fails)}")
