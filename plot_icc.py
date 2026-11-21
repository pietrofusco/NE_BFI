import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = pd.read_csv('icc_check_raw.csv')
k = ['tau', 'nonlin', 'n']
p = d[d.variant == 'paper']

# --- CSV tables
tab1 = p.groupby(k)[['tot_u', 'btw_u', 'btw_slope_dev', 'icc_hat', 'dR2_linear']].mean().round(3)
tab2 = d.pivot_table(index=k, columns='variant', values='dR2_linear').round(3)
tab1.to_csv('icc_decomposition.csv')
tab2.to_csv('icc_ablation.csv')

# --- figure: dR2 of BFI-Linear vs BFI-Hetero by variant (rows: tau, columns: nonlin)
taus, nls = sorted(d.tau.unique()), sorted(d.nonlin.unique())
lab = {'paper': 'paper DGM', 'no_intercept': 'tau = 0 (no intercept)',
       'intercept_only': 'intercept only (no slopes / covariate shift)'}
sty = {'paper': ('#2a78d6', 'o', '-'), 'no_intercept': ('#eb6834', 's', '--'),
       'intercept_only': ('#1baf7a', '^', ':')}
fig, axes = plt.subplots(len(taus), len(nls), figsize=(3.6 * len(nls), 2.9 * len(taus)),
                         sharex=True, sharey='row', squeeze=False)
for i, t in enumerate(taus):
    for j, nl in enumerate(nls):
        ax = axes[i][j]
        for v in ['paper', 'no_intercept', 'intercept_only']:
            g = d[(d.tau == t) & (d.nonlin == nl) & (d.variant == v)].groupby('n')['dR2_linear']
            m, se = g.mean(), g.std() / g.count()**0.5
            c, mk, ls = sty[v]
            ax.plot(m.index, m.values, color=c, marker=mk, ls=ls, lw=2, ms=4, label=lab[v])
            ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.15, lw=0)
        ax.axhline(0, color='k', lw=1)
        ax.set_xscale('log'); ax.set_xticks([5, 10, 20, 50, 100]); ax.set_xticklabels([5, 10, 20, 50, 100])
        ax.minorticks_off()
        ax.grid(True, color='#e4e3df')
        if i == 0: ax.set_title(f'non-linear share = {nl}', loc='left')
        if j == 0: ax.set_ylabel(f'$\\tau$ = {t}\n$\\Delta$R$^2$ BFI-Linear vs BFI-Hetero')
        if i == len(taus) - 1: ax.set_xlabel('patients per center')
handles, labels = axes[0][0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, frameon=False)
fig.savefig('icc_ablation.png', dpi=200, bbox_inches='tight')
print('Saved: icc_decomposition.csv, icc_ablation.csv, icc_ablation.png')
