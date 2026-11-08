#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reporting.py - tables and figures.

Colors: a fixed categorical palette assigned to the model (not to its rank), so each model keeps the
same color in every figure.  Each series also has its own marker/line style, so identity does not
rely on color alone.  Simulation heatmaps use a diverging map: two poles + grey midpoint.
"""
# %%

import pathlib

import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy  as np
import pandas as pd


# %% ---------------------------------------------------------------- style

PALETTE = [ '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948' ]
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

# model -> (color, marker, linestyle); models not listed get the free slots in a fixed order
_STYLE = {
    'BFI-Hetero'                                  : ( PALETTE[0], 'o', '-'  ),
    'Mixed-effects LMM'                           : ( INK2,       '',  ':'  ),
    'BFI-Linear'                                  : ( PALETTE[1], 's', '-'  ),
    'Ridge (pooled)'                              : ( PALETTE[1], '',  ':'  ),
    'Ridge (local)'                               : ( PALETTE[2], '^', '-'  ),
    'MLP (local)'                                 : ( PALETTE[2], 'v', '--' ),
    'MLP (pooled)'                                : ( PALETTE[7], 'D', '--' ),
    'Neural-BFI-Hetero [X|Z] FedProx'             : ( PALETTE[6], 'o', '-'  ),
    'Neural-BFI-Hetero [Z] FedProx'               : ( PALETTE[3], 's', '-'  ),
    'Neural-BFI-Hetero [X|Z] FedProx (unsup. AE)' : ( PALETTE[4], '^', '-'  ),
    'Neural-BFI-Hetero [X|Z] centralized'         : ( PALETTE[5], 'D', '--' ),
    # comparators (colors outside the base palette, distinct markers)
    'BFI-Fixed'                                   : ( '#8a5a2b',  'P', '-'  ),
    'DLMM'                                        : ( INK,        'x', '--' ),
    'Federated EM-LMM (iterative)'                : ( '#3aa6b9',  '*', '-.' ),
    'Ridge (local, tuned)'                        : ( PALETTE[2], '<', ':'  ),
}

POOLING_MODELS = [ 'Ridge (local)', 'BFI-Linear', 'BFI-Fixed', 'Mixed-effects LMM', 'BFI-Hetero' ]
NEURAL_MODELS  = [ 'Neural-BFI-Hetero [X|Z] FedProx', 'Neural-BFI-Hetero [Z] FedProx',
                   'Neural-BFI-Hetero [X|Z] FedProx (unsup. AE)', 'Neural-BFI-Hetero [X|Z] centralized' ]


def _style( model ):

    if model not in _STYLE:
        used = { v[0] for v in _STYLE.values() }
        free = [ c for c in PALETTE if c not in used ] or PALETTE
        _STYLE[model] = ( free[ len(_STYLE) % len(free) ], 'o', '-' )

    return _STYLE[model]


plt.rcParams.update( { 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                       'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 10,
                       'axes.spines.top': False, 'axes.spines.right': False,
                       'legend.frameon': False, 'savefig.bbox': 'tight' } )


def _grid( ax, axis = 'both' ):

    ax.grid( True, axis = axis, color = GRID, lw = 0.8 )
    ax.set_axisbelow( True )


def _p( out_dir,
        name ):

    out_dir = pathlib.Path( out_dir )
    out_dir.mkdir( parents = True, exist_ok = True )

    return out_dir / name


# %% ---------------------------------------------------------------- tables

def save_tables( summary,
                 paired,
                 out_dir,
                 header        = "",
                 communication = None ):

    summary.to_csv( _p( out_dir, 'cv_summary.csv' ), index = False )
    paired.to_csv(  _p( out_dir, 'cv_paired_comparisons.csv' ), index = False )
    if communication is not None:
        communication.to_csv( _p( out_dir, 'communication.csv' ), index = False )

    with open( _p( out_dir, 'Metrics_Performance.dat' ), 'w' ) as fid:

        fid.write( header + "\n" )
        fid.write( "--- Metrics (mean over repetitions; cluster-bootstrap 95% CI) ---\n" )
        w = summary['Model'].str.len().max()
        for _, r in summary.iterrows():

            fid.write( f"{r.Model:{w}s} : R2 = {r.R2:.3f} [{r.R2_lo:.3f}, {r.R2_hi:.3f}] - "
                       f"RMSE = {r.RMSE:.3f} [{r.RMSE_lo:.3f}, {r.RMSE_hi:.3f}] - "
                       f"MAE = {r.MAE:.3f} - CalibSlope = {r.CalibSlope:.3f}\n" )

        fid.write( "\n--- Paired comparisons vs reference ---\n" )
        for _, r in paired.iterrows():

            fid.write( f"{r.Model:{w}s} vs {r.Reference}: dR2 = {r.dR2:+.3f} [{r.dR2_lo:+.3f}, {r.dR2_hi:+.3f}] - "
                       f"p_boot = {r.p_bootstrap:.3f} - p_NB = {r.p_NB:.3f}\n" )

        if communication is not None:
            fid.write( "\n--- Communication (rounds beyond the common standardization round; "
                       "NaN = not federated) ---\n" )
            for _, r in communication.iterrows():
                conv = '' if not np.isfinite( r.converged ) else f" - converged in budget = {r.converged:.2f}"
                fid.write( f"{r.Model:{w}s} : rounds = {r.rounds_mean:.1f} [{r.rounds_min:.0f}, {r.rounds_max:.0f}]"
                           f" - floats per center per round = {r.msg_floats:.0f}{conv}\n" )


# %% ---------------------------------------------------------------- missing data

def _missing_text( s, by_feature, targets_csv = None ):
    """Readable text block for one dataset (used both per dataset and in the overall summary)."""

    L = [ f"=== {s['dataset']} ===" ]
    if 'n_rows_csv' in s:
        L.append( f"Rows in CSV: {s['n_rows_csv']} | dropped (target missing): {s['n_dropped_target_missing']} | "
                  f"dropped (cluster missing): {s['n_dropped_cluster_missing']} | "
                  f"dropped (clusters below min size): {s['n_dropped_small_clusters']}"
                  + ( f" [{s['small_clusters_removed']}]" if s.get( 'small_clusters_removed' ) else '' ) )
    L.append( f"Analysed: {s['n_subjects']} subjects, {s['n_clusters']} clusters, {s['n_features']} covariates" )
    L.append( f"Missing covariate cells: {s['pct_cells_missing']:.2f}%  "
              f"(continuous {s['pct_cells_missing_cont']:.2f}% | binary {s['pct_cells_missing_bin']:.2f}% | "
              f"z-score {s['pct_cells_missing_z']:.2f}%)".replace( 'nan%', 'n/a' ) )
    L.append( f"Complete cases: {s['n_complete_cases']} / {s['n_subjects']} ({s['pct_complete_cases']:.1f}%) | "
              f"distinct missingness patterns: {s['n_missing_patterns']}" )
    L.append( f"Clusters with any missing value: {s['n_clusters_with_missing']} / {s['n_clusters']} | "
              f"cluster x covariate cells entirely missing: {s['n_cluster_feature_all_missing']}" )
    if targets_csv is not None and len(targets_csv):
        L.append( "Original CSV, target/cluster: " +
                  ", ".join( f"{r.column} {r.pct_missing:.1f}% ({r.n_missing})" for r in targets_csv.itertuples() ) )
    w = max( 8, by_feature['feature'].str.len().max() )
    has_csv = 'pct_missing_csv' in by_feature.columns
    L.append( f"  {'feature':{w}s}  group  imputation      n_miss   %miss  " + ( "%miss_CSV  " if has_csv else "" ) +
              "clusters_any  clusters_all  %range_across_clusters" )
    for r in by_feature.itertuples():
        L.append( f"  {r.feature:{w}s}  {r.group:5s}  {r.imputation:14s}  {r.n_missing:6d}  {r.pct_missing:6.2f}  "
                  + ( f"{r.pct_missing_csv:9.2f}  " if has_csv else "" )
                  + f"{r.n_clusters_any_missing:12d}  {r.n_clusters_all_missing:12d}  "
                  f"[{r.pct_missing_cluster_min:.1f}, {r.pct_missing_cluster_max:.1f}]" )

    return "\n".join( L ) + "\n"


def save_missing_data( missing,
                       out_dir ):
    """
    Saves the missing-data report of one dataset (output of data_handler.missing_data_report):
      missing_summary.json, missing_by_feature.csv, missing_by_cluster.csv,
      missing_cluster_x_feature.csv, missing_patterns.csv, missing_targets_csv.csv, Missing_Data.dat
    """
    import json

    if not missing:
        return
    s = missing['summary']
    with open( _p( out_dir, 'missing_summary.json' ), 'w' ) as fid:
        json.dump( { k: ( None if isinstance( v, float ) and not np.isfinite( v ) else v ) for k, v in s.items() },
                   fid, indent = 2 )
    missing['by_feature'].to_csv( _p( out_dir, 'missing_by_feature.csv' ), index = False )
    missing['by_cluster'].to_csv( _p( out_dir, 'missing_by_cluster.csv' ), index = False )
    missing['cluster_x_feature'].to_csv( _p( out_dir, 'missing_cluster_x_feature.csv' ), index = False )
    missing['patterns'].to_csv( _p( out_dir, 'missing_patterns.csv' ), index = False )
    if missing.get( 'targets_csv' ) is not None:
        missing['targets_csv'].to_csv( _p( out_dir, 'missing_targets_csv.csv' ), index = False )

    with open( _p( out_dir, 'Missing_Data.dat' ), 'w' ) as fid:
        fid.write( _missing_text( s, missing['by_feature'], missing.get( 'targets_csv' ) ) )


def save_missing_summary( reports,
                          out_dir ):
    """
    Missing-data summary across all analysed datasets (reports: list of missing_data_report outputs).
      missing_data_summary.csv            : one row per dataset
      missing_data_by_feature_all.csv     : one row per (dataset, covariate)
      missing_data_summary.dat            : readable version (for the paper / supplementary material)
    """
    reports = [ r for r in reports if r ]
    if not reports:
        return None

    summ = pd.DataFrame( [ r['summary'] for r in reports ] )
    summ.to_csv( _p( out_dir, 'missing_data_summary.csv' ), index = False )
    pd.concat( [ r['by_feature'] for r in reports ], ignore_index = True ).to_csv(
        _p( out_dir, 'missing_data_by_feature_all.csv' ), index = False )

    with open( _p( out_dir, 'missing_data_summary.dat' ), 'w' ) as fid:
        fid.write( "MISSING DATA SUMMARY (covariates; rows without target or cluster are excluded before analysis)\n"
                   "Imputation inside each CV fold, fitted on the training part only: median (continuous, z-score), "
                   "most frequent (binary).\n"
                   "Simulated datasets (simulation study) have no missing values by construction.\n\n" )
        cols = [ 'dataset', 'n_subjects', 'n_clusters', 'n_features', 'pct_cells_missing', 'pct_complete_cases',
                 'n_features_with_missing', 'max_feature', 'max_feature_pct_missing' ]
        fid.write( summ[ [ c for c in cols if c in summ ] ].to_string( index = False, float_format = '%.2f' ) + "\n\n" )
        for r in reports:
            fid.write( _missing_text( r['summary'], r['by_feature'], r.get( 'targets_csv' ) ) + "\n" )

    return summ


# %% ---------------------------------------------------------------- CV on the datasets

def plot_forest( summary,
                 out_dir,
                 title = "" ):

    s = summary.sort_values( 'R2' )
    fig, axes = plt.subplots( 1, 2, figsize = (10, 0.38 * len(s) + 1.6), sharey = True )
    y = np.arange( len(s) )
    for ax, (m, lo, hi, lab) in zip( axes, [ ('R2', 'R2_lo', 'R2_hi', 'R$^2$'),
                                             ('RMSE', 'RMSE_lo', 'RMSE_hi', 'RMSE (outcome units)') ] ):

        for yi, (_, r) in zip( y, s.iterrows() ):
            c = _style( r.Model )[0]
            ax.errorbar( r[m], yi, xerr = [[ r[m] - r[lo] ], [ r[hi] - r[m] ]], fmt = 'o', ms = 5,
                         color = c, ecolor = c, elinewidth = 1.5, capsize = 0 )
        ax.set_xlabel( lab + '\n(mean over repeated CV; cluster-bootstrap 95% CI)' )
        _grid( ax, 'x' )

    axes[0].set_yticks( y )
    axes[0].set_yticklabels( s['Model'] )
    fig.suptitle( title, x = 0.02, ha = 'left', fontweight = 'bold' )
    fig.tight_layout()
    fig.savefig( _p( out_dir, 'forest_metrics.png' ), dpi = 200 )
    plt.close( fig )


def plot_paired( paired,
                 out_dir,
                 title = "" ):

    s   = paired.sort_values( 'dR2' )
    ref = s['Reference'].iloc[0]
    fig, ax = plt.subplots( figsize = (7, 0.38 * len(s) + 1.6) )
    y = np.arange( len(s) )
    for yi, (_, r) in zip( y, s.iterrows() ):
        c = _style( r.Model )[0]
        ax.errorbar( r.dR2, yi, xerr = [[ r.dR2 - r.dR2_lo ], [ r.dR2_hi - r.dR2 ]], fmt = 'o', ms = 5,
                     color = c, ecolor = c, elinewidth = 1.5, capsize = 0 )
    ax.axvline( 0, color = INK, lw = 1 )
    ax.set_yticks( y )
    ax.set_yticklabels( s['Model'] )
    ax.set_xlabel( f"$\\Delta$R$^2$ vs {ref}  (paired cluster-bootstrap 95% CI)\n"
                   f"< 0: worse than {ref}" )
    ax.set_title( title, loc = 'left', fontweight = 'bold' )
    _grid( ax, 'x' )
    fig.tight_layout()
    fig.savefig( _p( out_dir, 'paired_differences.png' ), dpi = 200 )
    plt.close( fig )


def plot_scatter( results,
                  out_dir ):

    models = list( results.preds.keys() )
    ncol   = min( 4, len(models) )
    nrow   = int( np.ceil( len(models) / ncol ) )
    fig, axes = plt.subplots( nrow, ncol, figsize = (3.2 * ncol, 3.2 * nrow), squeeze = False )
    y = results.y_true
    for ax, m in zip( axes.ravel(), models ):

        p = results.preds[m].mean( axis = 0 )
        ax.scatter( p, y, s = 5, alpha = 0.35, color = _style( m )[0], linewidths = 0 )
        lim = [ min( p.min(), y.min() ), max( p.max(), y.max() ) ]
        ax.plot( lim, lim, color = INK, ls = '--', lw = 1 )
        ax.set_title( m, fontsize = 8, loc = 'left' )
        ax.set_xlabel( 'predicted (mean over repetitions)' )
        ax.set_ylabel( 'observed' )
    for ax in axes.ravel()[len(models):]:
        ax.axis( 'off' )
    fig.tight_layout()
    fig.savefig( _p( out_dir, 'scatter_oof_predictions.png' ), dpi = 160 )
    plt.close( fig )


# %% ---------------------------------------------------------------- learning curves

def _lc_lines( ax, df, models, ycol, lo, hi ):

    for m in models:

        g = df[ df['Model'] == m ].sort_values( 'n_train_per_cluster' )
        if g.empty:
            continue
        c, mk, ls = _style( m )
        ax.plot( g['n_train_per_cluster'], g[ycol], color = c, marker = mk or None, ms = 5,
                 ls = ls, lw = 2, label = m )
        if m != 'Mixed-effects LMM':
            ax.fill_between( g['n_train_per_cluster'], g[lo], g[hi], color = c, alpha = 0.10, lw = 0 )


def plot_learning_curves( lc_summary,
                          lc_diff,
                          out_dir,
                          title = "" ):
    """
    Two figures:
      learning_curves.png        : partial pooling (local / homogeneous BFI / LMM / BFI-Hetero)
      learning_curves_neural.png : neural variants, dR2 vs reference
    """
    ref = lc_diff['Reference'].iloc[0]

    # --- partial pooling
    fig, axes = plt.subplots( 1, 2, figsize = (11, 4.2) )
    _lc_lines( axes[0], lc_summary, POOLING_MODELS, 'R2', 'R2_lo', 'R2_hi' )
    axes[0].set_ylabel( 'R$^2$ (repeated CV)' )
    axes[0].set_title( 'Predictive performance', loc = 'left' )
    _lc_lines( axes[1], lc_diff, [ m for m in POOLING_MODELS if m != ref ], 'dR2', 'dR2_lo', 'dR2_hi' )
    axes[1].axhline( 0, color = INK, lw = 1 )
    axes[1].set_ylabel( f"$\\Delta$R$^2$ vs {ref}" )
    axes[1].set_title( f'Paired difference vs {ref} (95% CI)', loc = 'left' )
    for ax in axes:
        ax.set_xlabel( 'training patients per center (mean)' )
        _grid( ax )
    axes[0].legend( loc = 'lower right' )
    fig.suptitle( f"{title}: local vs homogeneous vs partial pooling", x = 0.01, ha = 'left', fontweight = 'bold' )
    fig.tight_layout()
    fig.savefig( _p( out_dir, 'learning_curves.png' ), dpi = 200 )
    plt.close( fig )

    # --- neural
    nn = [ m for m in NEURAL_MODELS if m in set( lc_diff['Model'] ) ]
    if nn:
        fig, ax = plt.subplots( figsize = (6.5, 4.2) )
        _lc_lines( ax, lc_diff, nn, 'dR2', 'dR2_lo', 'dR2_hi' )
        ax.axhline( 0, color = INK, lw = 1 )
        ax.set_xlabel( 'training patients per center (mean)' )
        ax.set_ylabel( f"$\\Delta$R$^2$ vs {ref}  (95% CI)" )
        ax.set_title( f"{title}: neural extensions vs {ref}", loc = 'left', fontweight = 'bold' )
        _grid( ax )
        ax.legend( loc = 'lower right', fontsize = 8 )
        fig.tight_layout()
        fig.savefig( _p( out_dir, 'learning_curves_neural.png' ), dpi = 200 )
        plt.close( fig )


# %% ---------------------------------------------------------------- importance (appendix)

def plot_importance_stability( results,
                               model,
                               feature_names,
                               out_dir ):

    imps = [ e['importance'] for e in results.extras.get( model, [] ) if 'importance' in e ]
    if len(imps) == 0:
        return None

    imps = np.vstack( imps )
    mu, sd = imps.mean( 0 ), imps.std( 0, ddof = 1 )
    order  = np.argsort( mu )
    fig, ax = plt.subplots( figsize = (7, 0.35 * len(mu) + 1.6) )
    ax.barh( np.arange(len(mu)), mu[order], xerr = sd[order], color = PALETTE[0], height = 0.6,
             error_kw = { 'ecolor': INK2, 'elinewidth': 1 } )
    ax.set_yticks( np.arange(len(mu)) )
    ax.set_yticklabels( np.array(feature_names)[order] )
    ax.set_xlabel( 'Importance: mean |dy/dx| $\\times$ SD(x)  (mean $\\pm$ SD across folds)' )
    ax.set_title( f'{model}: importance stability ({len(imps)} folds)', loc = 'left' )
    _grid( ax, 'x' )
    fig.tight_layout()
    fig.savefig( _p( out_dir, 'gradient_importance_stability.png' ), dpi = 200 )
    plt.close( fig )

    pd.DataFrame( { 'feature': feature_names, 'mean': mu, 'sd': sd } ).sort_values(
        'mean', ascending = False ).to_csv( _p( out_dir, 'gradient_importance.csv' ), index = False )

    return mu


# %% ---------------------------------------------------------------- simulation

_DIVERGING = LinearSegmentedColormap.from_list( 'div', [ PALETTE[1], '#f2f1ee', PALETTE[0] ] )


def plot_simulation( summary,
                     paired,
                     out_dir,
                     summary_common = None,
                     neural         = 'Neural-BFI-Hetero [X|Z] FedProx',
                     linear         = ( 'BFI-Linear', 'BFI-Fixed', 'Mixed-effects LMM' ) ):
    """
    simulation_heatmap.png : dR2 (neural - reference) by tau x nonlin x n, with 1.96 MCSE and win rate
    simulation_paired.png  : dR2 vs reference (homogeneous BFI, LMM, neural) as n varies (95% MC bands)
    simulation_r2.png      : absolute R2 by model vs n (local ridge included), with the conditional ceiling.
                             Uses summary_common (same replicates for every model) if given, so the
                             curves are comparable; otherwise summary.
    """
    ref   = paired['reference'].iloc[0]
    taus  = sorted( paired['tau'].unique() )
    nls   = sorted( paired['nonlin'].unique() )
    ns    = sorted( paired['n_per_cluster'].unique() )
    icc   = summary.groupby( 'tau' )['icc_true'].mean()

    # ---------- neural heatmap
    d = paired[ paired['model'] == neural ]
    if not d.empty:
        vmax = max( 0.02, float( np.nanmax( np.abs( d['dR2'] ) ) ) )
        fig, axes = plt.subplots( 1, len(taus), figsize = (4.6 * len(taus) + 0.8, 3.6), squeeze = False )
        for ax, t in zip( axes[0], taus ):
            dt  = d[ d['tau'] == t ]
            piv = dt.pivot( index = 'nonlin', columns = 'n_per_cluster', values = 'dR2' ).reindex( index = nls, columns = ns )
            se  = dt.pivot( index = 'nonlin', columns = 'n_per_cluster', values = 'dR2_mcse' ).reindex( index = nls, columns = ns )
            wr  = dt.pivot( index = 'nonlin', columns = 'n_per_cluster', values = 'win_rate' ).reindex( index = nls, columns = ns )
            im  = ax.imshow( piv.values, cmap = _DIVERGING, vmin = -vmax, vmax = vmax, origin = 'lower', aspect = 'auto' )
            for i in range( len(nls) ):
                for j in range( len(ns) ):
                    v = piv.values[i, j]
                    if np.isfinite( v ):
                        col = 'white' if abs( v ) > 0.6 * vmax else INK
                        e   = se.values[i, j]
                        txt = f"{v:+.3f}" + ( f"\n±{1.96 * e:.3f}" if np.isfinite( e ) else "" ) + f"\nwin {wr.values[i, j]:.2f}"
                        ax.text( j, i, txt, ha = 'center', va = 'center', fontsize = 7, color = col )
            ax.set_xticks( range(len(ns)) ); ax.set_xticklabels( ns )
            ax.set_yticks( range(len(nls)) ); ax.set_yticklabels( nls )
            ax.set_xlabel( 'patients per center' )
            ax.set_title( f"$\\tau$ = {t}  (ICC $\\approx$ {icc.get( t, np.nan ):.2f})", loc = 'left' )
        axes[0][0].set_ylabel( 'non-linear share of signal' )
        cb = fig.colorbar( im, ax = axes[0].tolist(), shrink = 0.9 )
        cb.set_label( f"$\\Delta$R$^2$ vs {ref}\n(blue = neural better)" )
        n_rep = int( d['n_rep'].min() )
        fig.suptitle( f"Simulation: {neural} vs {ref}\n"
                      f"mean $\\Delta$R$^2$ ± 1.96 MCSE; win = P($\\Delta$R$^2$ > 0); {n_rep} datasets per cell",
                      x = 0.01, y = 1.08, ha = 'left', fontweight = 'bold', fontsize = 9 )
        fig.savefig( _p( out_dir, 'simulation_heatmap.png' ), dpi = 200 )
        plt.close( fig )

    # ---------- partial pooling vs n (panels: rows = tau, columns = nonlin)
    models = [ m for m in list( linear ) + [ neural ] if m in set( paired['model'] ) ]
    handles = {}
    fig, axes = plt.subplots( len(taus), len(nls), figsize = (3.6 * len(nls), 2.9 * len(taus)),
                              sharex = True, sharey = 'row', squeeze = False )
    for i, t in enumerate( taus ):
        for j, nl in enumerate( nls ):
            ax = axes[i][j]
            for m in models:
                g = paired[ (paired['tau'] == t) & (paired['nonlin'] == nl) & (paired['model'] == m) ].sort_values( 'n_per_cluster' )
                if g.empty:
                    continue
                c, mk, ls = _style( m )
                h, = ax.plot( g['n_per_cluster'], g['dR2'], color = c, marker = mk or 'o', ms = 4, ls = ls, lw = 2,
                              label = m )
                handles[m] = h
                ax.fill_between( g['n_per_cluster'], g['dR2'] - 1.96 * g['dR2_mcse'],
                                 g['dR2'] + 1.96 * g['dR2_mcse'], color = c, alpha = 0.15, lw = 0 )
            ax.axhline( 0, color = INK, lw = 1 )
            ax.set_xscale( 'log' ); ax.set_xticks( ns ); ax.set_xticklabels( ns )
            ax.minorticks_off()
            _grid( ax )
            if i == 0:
                ax.set_title( f"non-linear share = {nl}", loc = 'left' )
            if j == 0:
                ax.set_ylabel( f"$\\tau$ = {t} (ICC $\\approx$ {icc.get( t, np.nan ):.2f})\n$\\Delta$R$^2$ vs {ref}" )
            if i == len(taus) - 1:
                ax.set_xlabel( 'patients per center' )
    fig.suptitle( f"Simulation: paired $\\Delta$R$^2$ vs {ref} (bands: 95% Monte Carlo interval)",
                  x = 0.01, ha = 'left', fontweight = 'bold' )
    fig.tight_layout( rect = (0, 0.06, 1, 1) )
    fig.legend( list( handles.values() ), list( handles.keys() ), loc = 'lower center', ncol = len(handles), fontsize = 8 )
    fig.savefig( _p( out_dir, 'simulation_paired.png' ), dpi = 200 )
    plt.close( fig )

    # ---------- absolute R2 with conditional ceiling
    sa         = summary_common if summary_common is not None else summary
    models_abs = [ m for m in [ ref, 'Ridge (local)' ] + models if m in set( sa['model'] ) ]
    handles    = {}
    fig, axes = plt.subplots( len(taus), len(nls), figsize = (3.6 * len(nls), 2.9 * len(taus)),
                              sharex = True, sharey = True, squeeze = False )
    for i, t in enumerate( taus ):
        for j, nl in enumerate( nls ):
            ax = axes[i][j]
            for m in models_abs:
                g = sa[ (sa['tau'] == t) & (sa['nonlin'] == nl) & (sa['model'] == m) ].sort_values( 'n_per_cluster' )
                if g.empty:
                    continue
                c, mk, ls = _style( m )
                handles[m], = ax.plot( g['n_per_cluster'], g['R2'], color = c, marker = mk or 'o', ms = 4, ls = ls,
                                       lw = 2, label = m )
            g = sa[ (sa['tau'] == t) & (sa['nonlin'] == nl) & (sa['model'] == ref) ].sort_values( 'n_per_cluster' )
            handles['R$^2$ ceiling (seen centers)'], = ax.plot( g['n_per_cluster'], g['r2_max_cond'], color = INK2,
                                                                 ls = '--', lw = 1 )
            ax.set_xscale( 'log' ); ax.set_xticks( ns ); ax.set_xticklabels( ns )
            ax.minorticks_off()
            _grid( ax )
            if i == 0:
                ax.set_title( f"non-linear share = {nl}", loc = 'left' )
            if j == 0:
                ax.set_ylabel( f"$\\tau$ = {t}\nR$^2$ (CV)" )
            if i == len(taus) - 1:
                ax.set_xlabel( 'patients per center' )
    n_rep = int( sa['n_rep'].min() )
    fig.suptitle( f'Simulation: out-of-sample R$^2$ by model (same {n_rep} datasets for every model)',
                  x = 0.01, ha = 'left', fontweight = 'bold' )
    fig.tight_layout( rect = (0, 0.08, 1, 1) )
    fig.legend( list( handles.values() ), list( handles.keys() ), loc = 'lower center', ncol = 3, fontsize = 8 )
    fig.savefig( _p( out_dir, 'simulation_r2.png' ), dpi = 200 )
    plt.close( fig )


# %% ---------------------------------------------------------------- simulation tables for the paper

def save_simulation_tables( paired,
                            out_dir,
                            models = ( 'BFI-Fixed', 'BFI-Linear', 'DLMM', 'Federated EM-LMM (iterative)',
                                       'Ridge (local)', 'Ridge (local, tuned)' ) ):
    """
    For each model: a (tau, nonlin) x n_per_cluster table with 'dR2 (MCSE) [win]' vs the reference,
    ready to paste into the manuscript (e.g. BFI-Fixed vs BFI-Hetero shows where fixed intercepts
    overfit with small centers).  File: sim_table_<model>.csv
    """
    out = {}
    for m in models:

        d = paired[ paired['model'] == m ]
        if d.empty:
            continue
        cell = d.assign( txt = [ f"{a:+.3f} ({b:.3f}) [{w:.2f}]"
                                 for a, b, w in zip( d['dR2'], d['dR2_mcse'], d['win_rate'] ) ] )
        tab  = cell.pivot_table( index = [ 'tau', 'nonlin' ], columns = 'n_per_cluster', values = 'txt',
                                 aggfunc = 'first' )
        tag  = ''.join( ch if ch.isalnum() else '_' for ch in m )
        tab.to_csv( _p( out_dir, f'sim_table_{tag}.csv' ) )
        out[m] = tab

    return out


# %%
