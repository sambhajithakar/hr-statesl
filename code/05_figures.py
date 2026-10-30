"""Generate Figures 4 and 5 from the stored results. No value is typed by hand:
every plotted number is read from results/*.json or recomputed in place."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES, FIG = ROOT / 'results', ROOT / 'figures' / 'research-article'
FIG.mkdir(parents=True, exist_ok=True)
T = json.loads((RES / 'theory_and_simulation.json').read_text())
E = json.loads((RES / 'estimation_extended.json').read_text())
B = json.loads((RES / 'metric_blindness.json').read_text())

plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8,
                     'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight', 'savefig.dpi': 600})
C_COND, C_POOL, C_ACC = '#1b6ca8', '#c0392b', '#7f8c8d'

def panel_label(ax, s):
    ax.text(-0.18, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left')

# ------------------------------------------------------------------ Figure 4
fig = plt.figure(figsize=(7.2, 2.5))
gs = fig.add_gridspec(1, 3, wspace=0.42)

ax = fig.add_subplot(gs[0, 0])
r = np.linspace(0.05, 2.5, 400)
for pi1, ls in zip((0.50, 0.60, 0.70), ('-', '--', ':')):
    A = np.abs(pi1 - r * (1 - pi1)) / np.maximum(1.0, r)
    ax.plot(r, A, ls, color=C_COND, lw=1.4, label=f'$\\pi_1$ = {pi1:.2f}')
    rstar = pi1 / (1 - pi1)
    if rstar <= 2.5:
        ax.plot([rstar], [0], 'v', color=C_POOL, ms=5, clip_on=False)
ax.set_xlabel('reversal ratio $r$')
ax.set_ylabel('attenuation $A$ of the pooled effect')
ax.set_ylim(0, 0.72); ax.set_xlim(0, 2.5)
ax.legend(frameon=False, fontsize=7, loc='upper right')
ax.set_title('pooled estimand collapses at $r=\\pi_1/\\pi_2$', fontsize=8)
panel_label(ax, 'a')

ax = fig.add_subplot(gs[0, 1])
m = E['baseline']['measured_attenuation_discordant']
p = E['baseline']['predicted_attenuation_discordant']
ax.bar([0, 1], [p['mean'], m['mean']], yerr=[0, m['sd']], capsize=3, width=0.55,
       color=[C_ACC, C_COND], edgecolor='black', lw=0.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(['closed form', 'simulation'])
ax.set_ylabel('attenuation of discordant effect')
ax.set_ylim(0, 0.3)
for x, v in ((0, p['mean']), (1, m['mean'])):
    ax.text(x, v + 0.012, f'{v:.3f}', ha='center', fontsize=7)
ax.set_title('theory matches simulation', fontsize=8)
panel_label(ax, 'b')

ax = fig.add_subplot(gs[0, 2])
rs = [float(k.split('=')[1]) for k in E['sweep_reversal_ratio']]
pool = [E['sweep_reversal_ratio'][k]['pooled_discordant_median_rank']['mean']
        for k in E['sweep_reversal_ratio']]
cond = [E['sweep_reversal_ratio'][k]['state_conditioned_discordant_median_rank']['mean']
        for k in E['sweep_reversal_ratio']]
ax.plot(rs, pool, 'o-', color=C_POOL, lw=1.4, ms=4, label='pooled')
ax.plot(rs, cond, 's-', color=C_COND, lw=1.4, ms=4, label='state-conditioned')
ax.axhline(50, color='black', lw=0.7, ls='--')
ax.text(0.3, 58, 'selection cut-off $k$ = 50', fontsize=6.5)
ax.set_xlabel('reversal ratio $r$'); ax.set_ylabel('median rank of discordant genes')
ax.set_yscale('log'); ax.legend(frameon=False, fontsize=7, loc='upper left')
ax.set_title('discordant genes demoted by pooling', fontsize=8)
panel_label(ax, 'c')

for ext in ('svg', 'pdf', 'png'):
    fig.savefig(FIG / f'Figure1_pooling_attenuation.{ext}')
plt.close(fig)

# ------------------------------------------------------------------ Figure 5
fig = plt.figure(figsize=(7.2, 2.7))
gs = fig.add_gridspec(1, 3, wspace=0.52)

ax = fig.add_subplot(gs[0, 0])
vals = [B['pooled_precision_at_k']['mean'], B['state_conditioned_precision_at_k']['mean'],
        B['pooled_discordant_to_concordant_ratio']['mean'],
        B['state_conditioned_discordant_to_concordant_ratio']['mean']]
errs = [B['pooled_precision_at_k']['sd'], B['state_conditioned_precision_at_k']['sd'],
        B['pooled_discordant_to_concordant_ratio']['sd'],
        B['state_conditioned_discordant_to_concordant_ratio']['sd']]
x = [0, 0.75, 2.9, 3.65]
ax.bar(x, vals, yerr=errs, capsize=3, width=0.6,
       color=[C_POOL, C_COND, C_POOL, C_COND], edgecolor='black', lw=0.6)
ax.set_xlim(-0.6, 4.25)
ax.set_xticks([0.375, 3.275])
ax.set_xticklabels(['precision\n@ 50', 'discordant : concordant\ndetection ratio'], fontsize=7)
ax.set_ylabel('value')
ax.set_ylim(0, 1.6)
ax.text(0.375, 1.08, 'identical', ha='center', fontsize=7, style='italic')
ax.text(2.9, 0.06, '0.00', ha='center', fontsize=7, color=C_POOL)
ax.plot([], [], 's', color=C_POOL, label='pooled'); ax.plot([], [], 's', color=C_COND,
                                                            label='state-conditioned')
ax.legend(frameon=False, fontsize=7, loc='upper left')
ax.set_title('precision is blind to the loss', fontsize=8)
panel_label(ax, 'a')

ax = fig.add_subplot(gs[0, 1])
L = T['C_learning']['results']
labels = ['AUPRC\nstate 1', 'AUPRC\nstate 2', 'wrong-state\nhits @ 50']
keys = ['auprc_state1', 'auprc_state2', 'top_k_state2_wrong_state_hits']
w = 0.36; xs = np.arange(3)
for off, m, col in ((-w / 2, 'state_agnostic', C_POOL), (w / 2, 'state_conditioned', C_COND)):
    ax.bar(xs + off, [L[m][k]['mean'] for k in keys], w,
           yerr=[L[m][k]['sd'] for k in keys], capsize=2.5,
           color=col, edgecolor='black', lw=0.6,
           label='state-agnostic' if m == 'state_agnostic' else 'state-conditioned')
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel('value'); ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=7, loc='upper right')
ax.set_title('state covariate withheld', fontsize=8)
panel_label(ax, 'b')

ax = fig.add_subplot(gs[0, 2])
dj = T['D_design']['joint_80pct']
ds = [float(k.split('=')[1]) for k in dj]
ns = [dj[k]['n_replicates_per_arm'] for k in dj]
ax.plot(ds, ns, 'o-', color='#2d6a4f', lw=1.4, ms=5)
for xx, yy in zip(ds, ns):
    ax.annotate(str(yy), (xx, yy), textcoords='offset points', xytext=(5, 5), fontsize=7)
ax.set_xlabel('standardised per-state effect $|\\beta|/\\sigma$')
ax.set_ylabel('replicates per HR state')
ax.set_yscale('log'); ax.set_ylim(3.5, 80)
ax.set_yticks([5, 10, 20, 50]); ax.set_yticklabels(['5', '10', '20', '50'])
ax.minorticks_off()
ax.set_title('design for 80% joint power', fontsize=8)
panel_label(ax, 'c')

for ext in ('svg', 'pdf', 'png'):
    fig.savefig(FIG / f'Figure2_simulation_and_design.{ext}')
plt.close(fig)
print('figures written:', sorted(p.name for p in FIG.iterdir()))
