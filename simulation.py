#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation.py - simulation study, reported following ADEMP (Morris, White & Crowther, Stat Med 2019).

AIMS
  (1) how much federated partial pooling (BFI-Hetero) gains over local models and homogeneous BFI
      as the number of patients per center decreases and the between-center variance changes;
  (2) for which (n per center, non-linear share) the federated neural extractor improves on BFI-Hetero.

DATA-GENERATING MECHANISM (simulate_dataset)
  X : 7 continuous (AR(1) rho = 0.3 + center-specific mean shift), 3 binary (center-specific
      prevalence), 3 standard normal.
  y = 3 * ( sqrt(1-nonlin) f_lin(x) + sqrt(nonlin) f_nl(x) + u_c + eps ) + 10
      f_lin with center-specific slopes (slope_het); f_nl: sine, quadratic, interaction;
      u_c ~ N(0, tau^2); eps ~ N(0, 1/snr); f_lin and f_nl standardized.
  Factors: tau x nonlin x n_per_cluster (cluster sizes +-20%, minimum 5).

ESTIMAND      out-of-sample R^2 (5-fold CV stratified by center) and dR2 = R2(model) - R2(reference).
METHODS       cfg['simulation']['models_linear'] on n_datasets replicates; models_neural on the first
              n_datasets_neural (same datasets => paired comparisons).
PERFORMANCE   mean, MCSE = SD / sqrt(n_rep), win rate = P(dR2 > 0) with binomial MCSE.

Runs in parallel (joblib) and can be resumed: each scenario is saved in out_dir/parts/.  On a later
run, only the models missing from a saved scenario are computed, on the same datasets.
"""
# %%

import pathlib
import time

import numpy  as np
import pandas as pd


from data_handler import RawData
from evaluation   import CVRunner, build_registry, point_metrics


# %%

def simulate_dataset( n_clusters      = 25,
                      n_per_cluster   = 40,
                      nonlin          = 0.5,
                      tau             = 0.3,
                      slope_het       = 0.2,
                      covariate_shift = 0.3,
                      snr             = 1.0,
                      p_cont          = 7,
                      p_bin           = 3,
                      p_z             = 3,
                      min_size        = 5,
                      seed            = 0 ):

    rng   = np.random.default_rng( seed )
    p     = p_cont + p_bin + p_z
    sizes = np.maximum( min_size, np.round( n_per_cluster * rng.uniform( 0.8, 1.2, n_clusters ) ).astype(int) )

    cov   = 0.3 ** np.abs( np.subtract.outer( np.arange(p_cont), np.arange(p_cont) ) )
    L     = np.linalg.cholesky( cov )

    beta  = rng.normal( size = p )
    w1, w2, w3 = ( v / np.linalg.norm(v) for v in rng.normal( size = (3, p) ) )

    X_list, lin_list, cl_list, u_list = [], [], [], []
    for c in range(n_clusters):

        n      = sizes[c]
        shift  = rng.normal( 0, covariate_shift, p_cont )
        Xc     = np.empty( (n, p) )
        Xc[:, :p_cont]             = rng.normal( size = (n, p_cont) ) @ L.T + shift
        prev                       = 1.0 / (1.0 + np.exp( -0.5 * rng.normal( size = p_bin ) ))
        Xc[:, p_cont:p_cont+p_bin] = ( rng.uniform( size = (n, p_bin) ) < prev ).astype(float)
        Xc[:, p_cont+p_bin:]       = rng.normal( size = (n, p_z) )

        beta_c = beta + slope_het * rng.normal( size = p )
        X_list.append( Xc )
        lin_list.append( Xc @ beta_c )
        cl_list.append( np.full( n, c ) )
        u_list.append( np.full( n, rng.normal( 0, tau ) ) )

    X, lin = np.vstack(X_list), np.concatenate(lin_list)
    cl, u  = np.concatenate(cl_list), np.concatenate(u_list)

    Xs = ( X - X.mean(0) ) / ( X.std(0) + 1e-12 )
    nl = np.sin( 2.0 * Xs @ w1 ) + ( Xs @ w2 )**2 + np.tanh( Xs @ w3 ) * Xs[:, 0]

    lin_n = ( lin - lin.mean() ) / lin.std()
    nl_n  = ( nl  - nl.mean()  ) / nl.std()
    signal = np.sqrt(1.0 - nonlin) * lin_n + np.sqrt(nonlin) * nl_n

    eps = rng.normal( 0, np.sqrt(1.0 / snr), size = len(signal) )
    y   = 3.0 * ( signal + u + eps ) + 10.0

    names  = ( [f'cont{i}' for i in range(p_cont)] + [f'bin{i}' for i in range(p_bin)] +
               [f'z{i}' for i in range(p_z)] )
    groups = np.array( ['cont'] * p_cont + ['bin'] * p_bin + ['z'] * p_z )

    return RawData( X = X, Y = y.reshape(-1, 1), clusters = cl, feature_names = names,
                    groups = groups, target_names = ['y'],
                    name = f'sim_nl{nonlin}_n{n_per_cluster}_s{seed}',
                    info = { 'r2_max_pop' : float( np.var( 3.0 * signal ) / np.var( y ) ),
                             # conditional ceiling: for seen centers the center intercept is predictable
                             'r2_max_cond': float( np.var( 3.0 * ( signal + u ) ) / np.var( y ) ),
                             'icc_true'   : float( tau**2 / ( tau**2 + 1.0 / snr ) ),
                             'sizes'      : sizes.tolist() } )


# %%

def _scenario_seed( i_tau, i_nl, i_n, rep ):
    """Deterministic, distinct seed for each (scenario, replicate)."""

    return int( 10_000_000 + 1_000_000 * i_tau + 100_000 * i_nl + 10_000 * i_n + rep )


def _run_one_dataset( tau, nl, n_pc, rep, seed, sim, sim_cfg, names ):
    """One simulated dataset -> rows (one per model) with metrics and DGM information."""

    try:
        import torch
        torch.set_num_threads( 1 )
    except Exception:
        pass

    raw = simulate_dataset( n_clusters      = sim.get( 'n_clusters', 25 ),
                            n_per_cluster   = n_pc,
                            nonlin          = nl,
                            tau             = tau,
                            slope_het       = sim.get( 'slope_het', 0.2 ),
                            covariate_shift = sim.get( 'covariate_shift', 0.3 ),
                            snr             = sim.get( 'snr', 1.0 ),
                            seed            = seed )
    reg = build_registry( sim_cfg, only = names )
    res = CVRunner( raw, sim_cfg, reg ).run( verbose = False )

    rows = []
    for m, P in res.preds.items():

        pm = point_metrics( res.y_true, P )
        ex = res.extras[m]
        rows.append( { 'tau': tau, 'nonlin': nl, 'n_per_cluster': n_pc, 'rep': rep, 'seed': seed,
                       'model': m, 'R2': pm['R2'], 'RMSE': pm['RMSE'], 'CalibSlope': pm['CalibSlope'],
                       'tau2_hat' : float( np.mean( [ e['tau2'] for e in ex ] ) ) if 'tau2' in ex[0] else np.nan,
                       'r2_max_pop' : raw.info['r2_max_pop'], 'r2_max_cond': raw.info['r2_max_cond'],
                       'icc_true'   : raw.info['icc_true'],   'n_total'    : raw.n,
                       'time_sec'   : res.meta['timing_sec'][m],
                       # communication cost (rounds beyond standardization; NaN = not federated)
                       'comm_rounds': float( np.nanmean( [ e.get( 'comm_rounds', np.nan ) for e in ex ] ) )
                                      if any( 'comm_rounds' in e for e in ex ) else np.nan } )

    return rows


_KNOWN_ROUNDS = { 'Ridge (local)': 0.0, 'Ridge (local, tuned)': 0.0, 'BFI-Linear': 1.0, 'BFI-Hetero': 1.0,
                  'BFI-Fixed': 1.0, 'DLMM': 1.0 }


def _with_comm_rounds( df ):
    """For result files saved without the comm_rounds column: fill in the known deterministic values."""

    df = df.copy()
    if 'comm_rounds' not in df.columns:
        df['comm_rounds'] = np.nan
    miss = df['comm_rounds'].isna() & df['model'].isin( _KNOWN_ROUNDS )
    df.loc[miss, 'comm_rounds'] = df.loc[miss, 'model'].map( _KNOWN_ROUNDS )

    return df


def summarize_simulation( raw_df,
                          reference = 'BFI-Hetero' ):
    """
    -> (summary, paired)
       summary: per scenario x model -> n_rep, mean R2 and MCSE, mean CalibSlope
       paired : per scenario x model != reference, on the replicates where both were fitted
                -> mean dR2, MCSE, win rate and its MCSE
    """
    keys   = [ 'tau', 'nonlin', 'n_per_cluster' ]
    raw_df = _with_comm_rounds( raw_df )

    g = raw_df.groupby( keys + [ 'model' ] )
    summary = g.agg( n_rep        = ( 'R2', 'size' ),
                     R2           = ( 'R2', 'mean' ),
                     R2_sd        = ( 'R2', 'std' ),
                     CalibSlope   = ( 'CalibSlope', 'mean' ),
                     r2_max_pop   = ( 'r2_max_pop', 'mean' ),
                     r2_max_cond  = ( 'r2_max_cond', 'mean' ),
                     icc_true     = ( 'icc_true', 'mean' ),
                     time_sec     = ( 'time_sec', 'mean' ),
                     comm_rounds  = ( 'comm_rounds', 'mean' ) ).reset_index()
    summary['R2_mcse'] = summary['R2_sd'] / np.sqrt( summary['n_rep'] )

    ref  = raw_df[ raw_df['model'] == reference ][ keys + [ 'rep', 'R2' ] ].rename( columns = { 'R2': 'R2_ref' } )
    oth  = raw_df[ raw_df['model'] != reference ][ keys + [ 'rep', 'model', 'R2' ] ]
    m    = oth.merge( ref, on = keys + [ 'rep' ] )
    m['dR2'] = m['R2'] - m['R2_ref']
    m['win'] = ( m['dR2'] > 0 ).astype( float )

    paired = m.groupby( keys + [ 'model' ] ).agg( n_rep    = ( 'dR2', 'size' ),
                                                  dR2      = ( 'dR2', 'mean' ),
                                                  dR2_sd   = ( 'dR2', 'std' ),
                                                  win_rate = ( 'win', 'mean' ) ).reset_index()
    paired['dR2_mcse']      = paired['dR2_sd'] / np.sqrt( paired['n_rep'] )
    paired['win_rate_mcse'] = np.sqrt( paired['win_rate'] * ( 1 - paired['win_rate'] ) / paired['n_rep'] )
    paired['reference']     = reference

    return summary, paired


def run_simulation( cfg,
                    out_dir = None,
                    verbose = True ):
    """
    Runs the tau x nonlin x n_per_cluster grid.  With out_dir, each completed scenario is saved in
    out_dir/parts/ and skipped on later runs (resume after an interruption).
    -> (raw_df, summary, paired)
    """
    from joblib import Parallel, delayed

    sim       = cfg.get( 'simulation', {} )
    tau_grid  = sim.get( 'tau', [0.3] )
    tau_grid  = tau_grid if isinstance( tau_grid, (list, tuple) ) else [ tau_grid ]
    nl_grid   = sim.get( 'nonlin', [0.0, 0.3, 0.6] )
    n_grid    = sim.get( 'n_per_cluster', [5, 10, 20, 50, 100] )
    n_rep     = sim.get( 'n_datasets', 200 )
    n_rep_nn  = min( sim.get( 'n_datasets_neural', 50 ), n_rep )
    m_lin     = list( sim.get( 'models_linear', [ 'Ridge (local)', 'Ridge (pooled)', 'Mixed-effects LMM',
                                                  'BFI-Linear', 'BFI-Hetero', 'BFI-Fixed', 'DLMM',
                                                  'Federated EM-LMM (iterative)', 'Ridge (local, tuned)' ] ) )
    m_nn      = list( sim.get( 'models_neural', [ 'Neural-BFI-Hetero [X|Z] FedProx' ] ) )
    reference = sim.get( 'reference', 'BFI-Hetero' )
    n_jobs    = sim.get( 'n_jobs', -1 )

    sim_cfg = dict( cfg )
    sim_cfg['cv'] = { 'n_splits': 5, 'n_repeats': sim.get( 'n_cv_repeats', 1 ),
                      'seed': cfg.get( 'cv', {} ).get( 'seed', 0 ) }
    sim_cfg['latent_dim']         = sim.get( 'latent_dim', cfg.get( 'latent_dim', 8 ) )
    sim_cfg['compute_importance'] = False

    parts = None
    if out_dir is not None:
        parts = pathlib.Path( out_dir ) / 'parts'
        parts.mkdir( parents = True, exist_ok = True )

    all_rows = []
    for i_t, tau in enumerate( tau_grid ):
        for i_l, nl in enumerate( nl_grid ):
            for i_n, n_pc in enumerate( n_grid ):

                tag  = f"tau{tau}_nl{nl}_n{n_pc}"
                path = parts / f"{tag}.csv" if parts is not None else None

                # Incremental resume: if the scenario is already saved, only the missing models are
                # computed, on the same datasets (deterministic seeds and identical CV splits), and the
                # rows are appended.  Comparisons stay paired and the neural model is not rerun.
                df_old, miss_lin, miss_nn = None, list( m_lin ), list( m_nn )
                if path is not None and path.exists():
                    df_old   = _with_comm_rounds( pd.read_csv( path ) )
                    have     = set( df_old['model'] )
                    miss_lin = [ m for m in m_lin if m not in have ]
                    miss_nn  = [ m for m in m_nn  if m not in have ]
                    if not miss_lin and ( not miss_nn or n_rep_nn == 0 ):
                        if verbose:
                            print( f"  [sim] {tag}: already done ({df_old['rep'].nunique()} reps), skipped" )
                        all_rows.append( df_old )
                        continue

                t0   = time.time()
                jobs = [ delayed( _run_one_dataset )( tau, nl, n_pc, r, _scenario_seed( i_t, i_l, i_n, r ),
                                                      sim, sim_cfg, miss_lin + ( miss_nn if r < n_rep_nn else [] ) )
                         for r in range( n_rep ) if miss_lin or ( miss_nn and r < n_rep_nn ) ]
                out  = Parallel( n_jobs = n_jobs )( jobs )
                df_s = pd.DataFrame( [ row for rows in out for row in rows ] )
                if df_old is not None:
                    df_s = pd.concat( [ df_old, df_s ], ignore_index = True )
                if path is not None:
                    df_s.to_csv( path, index = False )
                all_rows.append( df_s )

                if verbose:
                    _, pr = summarize_simulation( df_s, reference )
                    msg = " | ".join( f"{r.model}: {r.dR2:+.3f}" for r in pr.itertuples() )
                    print( f"  [sim] {tag} ({time.time() - t0:.0f}s)  dR2 vs {reference}: {msg}" )

    raw_df = pd.concat( all_rows, ignore_index = True )
    summary, paired = summarize_simulation( raw_df, reference )

    return raw_df, summary, paired


# %%
