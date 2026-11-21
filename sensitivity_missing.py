#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensitivity_missing.py - missing covariate data: mechanism diagnostics and sensitivity analyses.

  * mcar_diagnostics(raw)          : compares subjects with and without missing covariates (outcome, center,
                                     covariates observed for everyone).  A non-significant test does NOT prove
                                     MCAR: it can only contradict it.
  * run_missing_sensitivity(...)   : repeats the repeated CV (same splits per scenario, same bootstrap samples
                                     per model) with alternative strategies and reports the dR2 contrasts with
                                     respect to the reference, i.e. the quantities the paper's conclusions rest on.

Scenarios (default):
    primary        median / mode (main analysis)
    complete_case  only subjects with complete covariates (CV and evaluation on this subsample)
    mean           mean / mode (federable in one round: data_handler.federated_mean_imputation_moments)
    indicators     median / mode + missingness indicators
    iterative      chained regressions (MAR given the observed covariates and the center)
    delta_<d>      median / mode, then the imputed values of the covariates in `delta_features` are shifted by
                   d x observed SD (MNAR)
"""
# %%

import numpy  as np
import pandas as pd

from dataclasses  import replace
from scipy        import stats

from evaluation   import CVRunner, build_registry, bootstrap_all, summarize, paired_comparisons


# %% ---------------------------------------------------- diagnostics

def _perm_chi2( groups, flag, n_perm, rng ):
    """Chi-square test of independence center x flag with permutation p-value (small expected cells)."""

    tab = pd.crosstab( groups, flag )
    chi = stats.chi2_contingency( tab, correction = False )[0]
    cnt = 0
    for _ in range( n_perm ):
        t = pd.crosstab( groups, rng.permutation( flag ) )
        cnt += stats.chi2_contingency( t, correction = False )[0] >= chi - 1e-12
    return float( chi ), int( tab.shape[0] - 1 ), float( ( cnt + 1 ) / ( n_perm + 1 ) )


def mcar_diagnostics( raw,
                      subsets = None,
                      n_perm  = 5000,
                      seed    = 0 ):
    """
    For each subset of 'incomplete' subjects (default: at least one missing covariate; if `subsets`
    is {name: [covariates]}, also the subjects with missing values in those covariates):
      - outcome: means, difference (incomplete - complete) with Welch 95% CI, Welch p-value;
      - center : chi-square center x incomplete with permutation p-value;
      - covariates observed for everyone: standardized mean difference (SMD) between incomplete and complete.
    -> (outcome_df, smd_df)
    """
    rng   = np.random.default_rng( seed )
    M     = np.isnan( raw.X )
    y     = raw.Y[:, 0]
    full  = np.where( ~M.any( axis = 0 ) )[0]
    defs  = { 'any covariate': np.arange( raw.X.shape[1] ) }
    for name, cols in ( subsets or {} ).items():
        defs[name] = [ raw.feature_names.index( c ) for c in cols ]

    rows, smd = [], []
    for name, cols in defs.items():

        inc        = M[:, cols].any( axis = 1 )
        a, b       = y[inc], y[~inc]
        t, p       = stats.ttest_ind( a, b, equal_var = False )
        va, vb     = a.var( ddof = 1 ) / len( a ), b.var( ddof = 1 ) / len( b )
        df         = ( va + vb )**2 / ( va**2 / ( len( a ) - 1 ) + vb**2 / ( len( b ) - 1 ) )
        diff, se   = a.mean() - b.mean(), np.sqrt( va + vb )
        q          = stats.t.ppf( 0.975, df )
        chi, dof, p_chi = _perm_chi2( raw.clusters, inc.astype( int ), n_perm, rng )
        rows.append( { 'incomplete_on': name, 'n_incomplete': int( inc.sum() ), 'n_complete': int( (~inc).sum() ),
                       'y_mean_incomplete': a.mean(), 'y_mean_complete': b.mean(),
                       'y_diff': diff, 'y_diff_lo': diff - q * se, 'y_diff_hi': diff + q * se, 'p_welch': float( p ),
                       'center_chi2': chi, 'center_df': dof, 'p_center_perm': p_chi } )

        for j in full:
            xa, xb = raw.X[inc, j], raw.X[~inc, j]
            sp     = np.sqrt( ( xa.var( ddof = 1 ) + xb.var( ddof = 1 ) ) / 2 )
            smd.append( { 'incomplete_on': name, 'covariate': raw.feature_names[j],
                          'mean_incomplete': xa.mean(), 'mean_complete': xb.mean(),
                          'smd': ( xa.mean() - xb.mean() ) / sp if sp > 0 else np.nan } )

    return pd.DataFrame( rows ), pd.DataFrame( smd )


# %% ---------------------------------------------------- sensitivity analysis

def default_scenarios( delta_features = ( 'cogcomp', 'langcomp', 'motcomp' ),
                       deltas         = ( -1.0, -0.5, 0.5, 1.0 ) ):

    sc = { 'primary'       : { 'imputation': 'median' },
           'complete_case' : { 'imputation': 'median', '_complete_case': True },
           'mean'          : { 'imputation': 'mean' },
           'indicators'    : { 'imputation': 'median', 'indicators': True },
           'iterative'     : { 'imputation': 'iterative' } }
    for d in deltas:
        sc[f'delta_{d:+.1f}'] = { 'imputation': 'median', 'delta': { f: d for f in delta_features } }

    return sc


def complete_cases( raw ):
    """RawData restricted to subjects without missing covariates (same centers, same columns)."""

    keep = ~np.isnan( raw.X ).any( axis = 1 )

    return replace( raw, X = raw.X[keep], Y = raw.Y[keep], clusters = raw.clusters[keep],
                    name = f"{raw.name} (complete cases)", missing = {} )


def run_missing_sensitivity( raw,
                             cfg,
                             models    = None,
                             scenarios = None,
                             reference = 'BFI-Hetero',
                             verbose   = True ):
    """
    -> (summary_long, paired_long): one row per (scenario, model), with R2 and CI, and dR2 vs reference and CI.
    models: registry names (default: cfg['models'], i.e. the non-neural models of the real datasets).
    """
    cv        = cfg.get( 'cv', {} )
    models    = list( models if models is not None else cfg.get( 'models', [] ) )
    if reference not in models:
        models.append( reference )
    scenarios = scenarios if scenarios is not None else default_scenarios()

    summ_all, pair_all = [], []
    for name, opts in scenarios.items():

        opts  = dict( opts )
        cc    = opts.pop( '_complete_case', False )
        data  = complete_cases( raw ) if cc else raw
        c     = dict( cfg, preprocessing = opts )
        reg   = build_registry( c, only = models )
        if verbose:
            print( f"  [missing sensitivity] {name}: n = {data.n}, options = {opts}" )

        res   = CVRunner( data, c, reg ).run( verbose = False )
        boot  = bootstrap_all( res, n_boot = cv.get( 'n_boot', 2000 ), unit = cv.get( 'boot_unit', 'cluster' ),
                               seed = cv.get( 'seed', 0 ) )
        s     = summarize( res, boot, ci = cv.get( 'ci', 0.95 ) )
        p     = paired_comparisons( res, boot, reference, ci = cv.get( 'ci', 0.95 ) )
        for df in ( s, p ):
            df.insert( 0, 'scenario', name )
            df.insert( 1, 'n', data.n )
        summ_all.append( s )
        pair_all.append( p )

    return pd.concat( summ_all, ignore_index = True ), pd.concat( pair_all, ignore_index = True )


def sensitivity_table( paired_long,
                       models = None ):
    """Wide model x scenario table with 'dR2 [lo, hi]' (ready for the supplementary)."""

    d = paired_long if models is None else paired_long[ paired_long['Model'].isin( models ) ]
    d = d.assign( txt = [ f"{a:+.3f} [{l:+.3f}, {h:+.3f}]" for a, l, h in zip( d['dR2'], d['dR2_lo'], d['dR2_hi'] ) ] )
    order = list( dict.fromkeys( paired_long['scenario'] ) )

    return d.pivot( index = 'Model', columns = 'scenario', values = 'txt' )[ order ]


# %%
