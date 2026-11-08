#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py - repeated CV, bootstrap, paired comparisons, learning curves.

  * build_registry(cfg)   : {model_name: callable(fold, seed) -> {'pred','extras'}}
  * CVRunner.run(...)     : repeated k-fold stratified by cluster; on every fold all models see the
                            same data (paired comparisons); preprocessing fitted on the fold's training
                            part only; predictions mapped back to the original scale.
  * bootstrap_all / summarize / paired_comparisons / nadeau_bengio_test
                          : bootstrap CIs over clusters (same resamples for every model).
  * run_learning_curve    : repeated CV with the training set subsampled within clusters.
  * communication_summary : communication rounds and message size per model.

cfg['models'] (optional) restricts the models run on the real datasets.

The variability across repetitions measures sensitivity to the split, not sampling uncertainty:
CIs are obtained by resampling subjects/clusters.  Evaluation supports a single target (q = 1) only.
"""


# %%

import time
import numpy  as np
import pandas as pd


from dataclasses  import dataclass
from dataclasses  import field
from scipy        import stats
from data_handler import make_repeated_cv_splits
from data_handler import build_fold
from models       import BaselineModels
from models       import fit_predict_bfi_linear
from models       import fit_predict_neural_bfi
from models       import fit_predict_bfi_fixed
from models       import fit_predict_dlmm
from models       import fit_predict_fed_em_lmm


# %%

@dataclass
class CVResults:

    y_true    : np.ndarray
    clusters  : np.ndarray
    preds     : dict
    fold_mse  : dict
    n_train   : np.ndarray
    n_test    : np.ndarray
    extras    : dict = field( default_factory = dict )
    meta      : dict = field( default_factory = dict )


# %%

def point_metrics( y,
                   P ):
    """y: (N,), P: (R,N) -> dict of metrics averaged over the R repetitions."""

    err  = P - y
    sst  = np.sum( (y - y.mean())**2 )
    mse  = np.mean( err**2, axis = 1 )
    mae  = np.mean( np.abs(err), axis = 1 )
    r2   = 1.0 - np.sum( err**2, axis = 1 ) / sst

    # calibration slope: regression of y on y_hat (ideal = 1)
    slopes = []
    for p in P:

        vp = np.var(p)
        slopes.append( np.cov( p, y, bias = True )[0, 1] / vp if vp > 1e-12 else np.nan )

    return { 'MSE'        : float(mse.mean()),
             'RMSE'       : float(np.sqrt(mse).mean()),
             'MAE'        : float(mae.mean()),
             'R2'         : float(r2.mean()),
             'CalibSlope' : float(np.nanmean(slopes)),
             'R2_sd_rep'  : float(r2.std(ddof=1)) if len(r2) > 1 else 0.0 }


def _resample_indices( clusters,
                       n_boot,
                       unit,
                       rng ):

    N = len(clusters)
    if unit == 'subject':

        return [ rng.integers( 0, N, N ) for _ in range(n_boot) ]

    uc      = np.unique( clusters )
    members = { c: np.where(clusters == c)[0] for c in uc }

    return [ np.concatenate( [ members[c] for c in rng.choice( uc, size = len(uc), replace = True ) ] )
             for _ in range(n_boot) ]


def _boot_metric_matrix( y,
                         P,
                         idx_list ):
    """-> array (B, 3): MSE, MAE, R2, averaged over repetitions, for each bootstrap sample."""

    out = np.empty( (len(idx_list), 3) )
    for b, idx in enumerate(idx_list):

        yb        = y[idx]
        err       = P[:, idx] - yb
        sst       = np.sum( (yb - yb.mean())**2 )
        out[b, 0] = np.mean( err**2 )
        out[b, 1] = np.mean( np.abs(err) )
        out[b, 2] = np.mean( 1.0 - np.sum( err**2, axis = 1 ) / sst )

    return out


def bootstrap_all( results,
                   n_boot = 2000,
                   unit   = 'cluster',
                   seed   = 0 ):
    """Returns {model: (B,3)} using the same resamples for every model."""

    rng      = np.random.default_rng( seed )
    idx_list = _resample_indices( results.clusters, n_boot, unit, rng )

    return { m: _boot_metric_matrix( results.y_true, P, idx_list )
             for m, P in results.preds.items() }


# %%

def summarize( results,
               boot,
               ci = 0.95 ):
    """Table with one row per model: metrics and CIs (bootstrap percentiles)."""

    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    rows   = []
    for m, P in results.preds.items():

        pm  = point_metrics( results.y_true, P )
        B   = boot[m]
        mse_lo, mse_hi = np.percentile( B[:, 0], [lo, hi] )
        mae_lo, mae_hi = np.percentile( B[:, 1], [lo, hi] )
        r2_lo,  r2_hi  = np.percentile( B[:, 2], [lo, hi] )

        rows.append( { 'Model'      : m,
                       'R2'         : pm['R2'],   'R2_lo'   : r2_lo,  'R2_hi'   : r2_hi,
                       'RMSE'       : pm['RMSE'], 'RMSE_lo' : np.sqrt(mse_lo), 'RMSE_hi' : np.sqrt(mse_hi),
                       'MAE'        : pm['MAE'],  'MAE_lo'  : mae_lo, 'MAE_hi'  : mae_hi,
                       'CalibSlope' : pm['CalibSlope'],
                       'R2_sd_rep'  : pm['R2_sd_rep'] } )

    return pd.DataFrame( rows )


def nadeau_bengio_test( mse_a,
                        mse_b,
                        n_train,
                        n_test ):
    """
    Paired t-test on per-fold MSEs with the Nadeau & Bengio (2003) variance correction.
    Heuristic for repeated k-fold: per-fold estimates are not independent (overlapping training
    sets), and the (1/J + n_test/n_train) correction avoids overly optimistic p-values.
    """
    d = np.asarray(mse_a) - np.asarray(mse_b)
    J = len(d)
    if J < 3 or np.var(d, ddof = 1) < 1e-16:
        return np.nan, np.nan

    corr = 1.0 / J + np.mean(n_test) / np.mean(n_train)
    t    = d.mean() / np.sqrt( corr * np.var(d, ddof = 1) )

    return float(t), float( 2 * stats.t.sf( abs(t), df = J - 1 ) )


def paired_comparisons( results,
                        boot,
                        reference,
                        ci = 0.95 ):
    """
    Paired comparisons (model - reference) on MSE and R2.
    R2 difference > 0 => the model beats the reference; same for MSE difference < 0.
    """
    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    rows   = []
    for m in results.preds.keys():

        if m == reference:
            continue

        d_mse = boot[m][:, 0] - boot[reference][:, 0]
        d_r2  = boot[m][:, 2] - boot[reference][:, 2]
        B     = len(d_r2)
        p_boot = min( 1.0, 2 * min( np.mean(d_r2 <= 0), np.mean(d_r2 >= 0) ) )
        p_boot = max( p_boot, 1.0 / B )

        pm_m = point_metrics( results.y_true, results.preds[m] )
        pm_r = point_metrics( results.y_true, results.preds[reference] )
        t, p_nb = nadeau_bengio_test( results.fold_mse[m], results.fold_mse[reference],
                                      results.n_train, results.n_test )

        rows.append( { 'Model'       : m,
                       'Reference'   : reference,
                       'dR2'         : pm_m['R2'] - pm_r['R2'],
                       'dR2_lo'      : np.percentile( d_r2, lo ),
                       'dR2_hi'      : np.percentile( d_r2, hi ),
                       'dMSE'        : pm_m['MSE'] - pm_r['MSE'],
                       'dMSE_lo'     : np.percentile( d_mse, lo ),
                       'dMSE_hi'     : np.percentile( d_mse, hi ),
                       'p_bootstrap' : p_boot,
                       't_NB'        : t,
                       'p_NB'        : p_nb } )

    return pd.DataFrame( rows )


# %%


def variance_components( results ):
    """
    Per-fold variance components (standardized y scale) from the models that estimate them:
    BFI-Hetero and its variants (extras 'tau2','sigma2'), LMM (extras 'var_re','sigma2').
    ICC = tau2 / (tau2 + sigma2): share of residual variance due to the center.
    -> DataFrame: Model, n_folds, tau2, sigma2, ICC (mean), ICC_lo/ICC_hi (2.5/97.5 percentiles over folds).
    """
    rows = []
    for m, ex in results.extras.items():

        t2 = [ e.get( 'tau2', e.get( 'var_re' ) ) for e in ex ]
        s2 = [ e.get( 'sigma2' ) for e in ex ]
        ok = [ (a, b) for a, b in zip( t2, s2 ) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) ]
        if not ok or not ( 'tau2' in ex[0] or 'var_re' in ex[0] ):
            continue

        t2, s2 = np.array( ok ).T
        icc    = t2 / ( t2 + s2 )
        rows.append( { 'Model': m, 'n_folds': len(icc), 'tau2': t2.mean(), 'sigma2': s2.mean(),
                       'ICC': icc.mean(), 'ICC_lo': np.percentile( icc, 2.5 ), 'ICC_hi': np.percentile( icc, 97.5 ) } )

    return pd.DataFrame( rows )


def communication_summary( results ):
    """
    Communication cost per model, from the fold extras:
      comm_rounds : rounds beyond the common standardization round (0 = local model; NaN = not federated)
      msg_floats  : values sent by one center per round (message size)
      converged   : share of folds in which the iterative algorithm reached the tolerance within budget
    -> DataFrame: Model, federated, rounds_mean, rounds_min, rounds_max, msg_floats, converged
    """
    rows = []
    for m, ex in results.extras.items():

        r = np.array( [ e.get( 'comm_rounds', np.nan ) for e in ex ], dtype = float )
        f = np.array( [ e.get( 'msg_floats',  np.nan ) for e in ex ], dtype = float )
        c = np.array( [ e.get( 'converged',   np.nan ) for e in ex ], dtype = float )
        fed = bool( np.isfinite( r ).any() and np.nanmax( r ) > 0 )
        rows.append( { 'Model'       : m,
                       'federated'   : fed,
                       'rounds_mean' : float( np.nanmean( r ) ) if np.isfinite( r ).any() else np.nan,
                       'rounds_min'  : float( np.nanmin( r ) )  if np.isfinite( r ).any() else np.nan,
                       'rounds_max'  : float( np.nanmax( r ) )  if np.isfinite( r ).any() else np.nan,
                       'msg_floats'  : float( np.nanmean( f ) ) if np.isfinite( f ).any() else np.nan,
                       'converged'   : float( np.nanmean( c ) ) if np.isfinite( c ).any() else np.nan } )

    return pd.DataFrame( rows )


# %% ---------------------------------------------------- CV runner

# Neural variants are federated by default; 'hetero' = BFI with center intercepts.
DEFAULT_NEURAL_VARIANTS = {
    'Neural-BFI-Hetero [X|Z] FedProx' : { 'skip': True,  'federated': True,  'hetero': True },
    'Neural-BFI-Hetero [Z] FedProx'   : { 'skip': False, 'federated': True,  'hetero': True },
}


def build_registry( cfg,
                    only = None ):

    bm = BaselineModels( alpha        = cfg.get( 'ridge_alpha', cfg.get('lambda_local', 1.0) ),
                         mlp_alpha    = cfg.get( 'mlp_alpha', 1.0 ),
                         mlp_hidden   = cfg.get( 'mlp_hidden', (32,) ),
                         mlp_max_iter = cfg.get( 'mlp_max_iter', 1000 ) )

    reg = { 'Ridge (local)'      : lambda fold, seed: bm.local_ridge( fold ),
            'Ridge (pooled)'     : lambda fold, seed: bm.pooled_ridge( fold ),
            'MLP (local)'        : lambda fold, seed: bm.local_mlp( fold, seed ),
            'MLP (pooled)'       : lambda fold, seed: bm.pooled_mlp( fold, seed ),
            'Mixed-effects LMM'  : lambda fold, seed: bm.mixed_effects( fold ),
            'BFI-Linear'         : lambda fold, seed: fit_predict_bfi_linear( fold, cfg ),
            'BFI-Hetero'         : lambda fold, seed: fit_predict_bfi_linear( fold, cfg, hetero = True ),
            # comparators and sensitivity analysis
            'BFI-Fixed'                    : lambda fold, seed: fit_predict_bfi_fixed( fold, cfg ),
            'DLMM'                         : lambda fold, seed: fit_predict_dlmm( fold, cfg ),
            'Federated EM-LMM (iterative)' : lambda fold, seed: fit_predict_fed_em_lmm( fold, cfg ),
            'Ridge (local, tuned)'         : lambda fold, seed: bm.local_ridge_tuned( fold ) }

    # subset of models to run on the real datasets (None = all)
    if only is None and cfg.get( 'models' ) is not None:
        only = list( cfg['models'] ) + list( cfg.get( 'neural_variants', DEFAULT_NEURAL_VARIANTS ).keys() )

    variants = dict( cfg.get( 'neural_variants', DEFAULT_NEURAL_VARIANTS ) )

    for name, spec in variants.items():

        reg[name] = ( lambda fold, seed, s = spec:
                      fit_predict_neural_bfi( fold, cfg, seed,
                                              skip_connection = s.get( 'skip', True ),
                                              federated       = s.get( 'federated', True ),
                                              neural_override = s.get( 'neural', None ),
                                              hetero          = s.get( 'hetero', False ) ) )

    if only is not None:

        reg = { k: v for k, v in reg.items() if k in only }

    return reg


# %%

class CVRunner:

    def __init__( self,
                  raw,
                  cfg,
                  registry ):

        if raw.Y.shape[1] != 1:
            raise ValueError( "CV evaluation supports a single target only (q = 1)." )

        self.raw      = raw
        self.cfg      = cfg
        self.registry = registry


    def run( self,
             train_fraction = 1.0,
             n_repeats      = None,
             n_splits       = None,
             verbose        = True ):

        cv        = self.cfg.get( 'cv', {} )
        n_splits  = n_splits  if n_splits  is not None else cv.get( 'n_splits', 5 )
        n_repeats = n_repeats if n_repeats is not None else cv.get( 'n_repeats', 10 )
        seed      = cv.get( 'seed', 0 )

        raw    = self.raw
        y_true = raw.Y[:, 0].copy()
        splits = make_repeated_cv_splits( raw.clusters, n_splits, n_repeats, seed )

        preds    = { m: np.full( (n_repeats, raw.n), np.nan ) for m in self.registry }
        fold_mse = { m: [] for m in self.registry }
        extras   = { m: [] for m in self.registry }
        timing   = { m: 0.0 for m in self.registry }
        n_train, n_test = [], []

        t0 = time.time()
        for s in splits:

            r, k      = s['repeat'], s['fold']
            fold_seed = seed + 7919 * r + k
            fold      = build_fold( raw, s['train_idx'], s['test_idx'],
                                    train_fraction = train_fraction, seed = fold_seed )
            y_te      = fold['pooled_test']['Y_raw'][:, 0]
            n_train.append( len(fold['pooled_train']['idx']) )
            n_test.append( len(s['test_idx']) )

            for name, fn in self.registry.items():

                t1  = time.time()
                out = fn( fold, fold_seed )
                timing[name] += time.time() - t1

                pred_real = fold['preproc'].inverse_Y( np.asarray(out['pred']).reshape(-1, 1) )[:, 0]
                preds[name][r, s['test_idx']] = pred_real
                fold_mse[name].append( float( np.mean( (pred_real - y_te)**2 ) ) )
                extras[name].append( { 'repeat': r, 'fold': k, **out.get('extras', {}) } )

            if verbose and k == n_splits - 1:

                print( f"  [{raw.name}] repeat {r + 1}/{n_repeats} "
                       f"(frac={train_fraction:.2f}) - {time.time() - t0:.0f}s" )

        for m, P in preds.items():

            if np.isnan(P).any():
                raise RuntimeError( f"Missing predictions for model {m}" )

        return CVResults( y_true   = y_true,
                          clusters = raw.clusters,
                          preds    = preds,
                          fold_mse = { m: np.array(v) for m, v in fold_mse.items() },
                          n_train  = np.array(n_train),
                          n_test   = np.array(n_test),
                          extras   = extras,
                          meta     = { 'n_splits': n_splits, 'n_repeats': n_repeats,
                                       'train_fraction': train_fraction,
                                       'mean_train_per_cluster':
                                           float( np.mean(n_train) / len(np.unique(raw.clusters)) ),
                                       'timing_sec': timing } )


# %% ---------------------------------------------------- learning curves

def run_learning_curve( raw,
                        cfg,
                        registry,
                        reference = 'BFI-Hetero',
                        verbose   = True ):

    lc        = cfg.get( 'learning_curve', {} )
    fractions = lc.get( 'fractions', [0.25, 0.5, 0.75, 1.0] )
    n_repeats = lc.get( 'n_repeats', 3 )
    n_boot    = lc.get( 'n_boot', 500 )
    unit      = cfg.get( 'cv', {} ).get( 'boot_unit', 'cluster' )
    seed      = cfg.get( 'cv', {} ).get( 'seed', 0 )

    summaries, diffs = [], []
    for f in fractions:

        if verbose:
            print( f"[{raw.name}] learning curve: training fraction = {f:.2f}" )

        res  = CVRunner( raw, cfg, registry ).run( train_fraction = f,
                                                   n_repeats      = n_repeats,
                                                   verbose        = False )
        boot = bootstrap_all( res, n_boot = n_boot, unit = unit, seed = seed )

        s = summarize( res, boot )
        s['fraction']            = f
        s['n_train_per_cluster'] = res.meta['mean_train_per_cluster']
        summaries.append( s )

        d = paired_comparisons( res, boot, reference )
        d['fraction']            = f
        d['n_train_per_cluster'] = res.meta['mean_train_per_cluster']
        diffs.append( d )

    return pd.concat( summaries, ignore_index = True ), pd.concat( diffs, ignore_index = True )


# %%
