#!/usr/bin/env python3
"""
Which component of the between-center variance explains the gain of BFI-Hetero over BFI-Linear?

(a) decomposition of the variance of y in the generator (exact replica of simulation.simulate_dataset,
    checked line by line) into: common signal (X beta), deviation due to center-specific slopes
    (X (beta_c - beta)), non-linear part, intercept u, noise; and the BETWEEN-center share of each;
(b) CV identical to the one in the paper (5-fold stratified by center, 1 repetition) for BFI-Linear and
    BFI-Hetero, with the ICC estimated by BFI-Hetero;
(c) ablations on the same seeds: tau = 0 (no intercept) and slope_het = covariate_shift = 0
    (intercept only).
"""
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from simulation import simulate_dataset, scenario_seed
from evaluation import CVRunner, build_registry, point_metrics


def components( n_clusters=25, n_per_cluster=40, nonlin=0.5, tau=0.3, slope_het=0.2, covariate_shift=0.3,
                snr=1.0, p_cont=7, p_bin=3, p_z=3, min_size=5, seed=0 ):
    """Copy of simulate_dataset that also returns the components (same RNG stream)."""
    rng   = np.random.default_rng( seed )
    p     = p_cont + p_bin + p_z
    sizes = np.maximum( min_size, np.round( n_per_cluster * rng.uniform( 0.8, 1.2, n_clusters ) ).astype(int) )
    cov   = 0.3 ** np.abs( np.subtract.outer( np.arange(p_cont), np.arange(p_cont) ) )
    L     = np.linalg.cholesky( cov )
    beta  = rng.normal( size = p )
    w1, w2, w3 = ( v / np.linalg.norm(v) for v in rng.normal( size = (3, p) ) )
    X_l, com_l, dev_l, cl_l, u_l = [], [], [], [], []
    for c in range(n_clusters):
        n = sizes[c]
        shift = rng.normal( 0, covariate_shift, p_cont )
        Xc = np.empty( (n, p) )
        Xc[:, :p_cont] = rng.normal( size = (n, p_cont) ) @ L.T + shift
        prev = 1.0 / (1.0 + np.exp( -0.5 * rng.normal( size = p_bin ) ))
        Xc[:, p_cont:p_cont+p_bin] = ( rng.uniform( size = (n, p_bin) ) < prev ).astype(float)
        Xc[:, p_cont+p_bin:] = rng.normal( size = (n, p_z) )
        beta_c = beta + slope_het * rng.normal( size = p )
        X_l.append( Xc ); com_l.append( Xc @ beta ); dev_l.append( Xc @ ( beta_c - beta ) )
        cl_l.append( np.full( n, c ) ); u_l.append( np.full( n, rng.normal( 0, tau ) ) )
    X = np.vstack( X_l ); com = np.concatenate( com_l ); dev = np.concatenate( dev_l )
    cl = np.concatenate( cl_l ); u = np.concatenate( u_l )
    lin = com + dev
    Xs = ( X - X.mean(0) ) / ( X.std(0) + 1e-12 )
    nl = np.sin( 2.0 * Xs @ w1 ) + ( Xs @ w2 )**2 + np.tanh( Xs @ w3 ) * Xs[:, 0]
    sl, sn = lin.std(), nl.std()
    a, b = np.sqrt( 1 - nonlin ) / sl, np.sqrt( nonlin ) / sn
    eps = rng.normal( 0, np.sqrt( 1.0 / snr ), size = len( lin ) )
    parts = { 'common_lin': 3 * a * ( com - com.mean() ), 'slope_dev': 3 * a * ( dev - dev.mean() ),
              'nonlin': 3 * b * ( nl - nl.mean() ), 'u': 3 * u, 'eps': 3 * eps }
    y = sum( parts.values() ) + 3 * a * ( lin.mean() - lin.mean() ) + 10 + 3 * u.mean() * 0
    return X, y, cl, parts


def between( v, cl ):
    m = pd.Series( v ).groupby( cl ).transform( 'mean' ).values
    return m


def one( tau, nl, n, rep, variant ):
    seed = scenario_seed( tau, nl, n, rep )
    kw   = dict( n_clusters = 25, n_per_cluster = n, nonlin = nl, tau = tau, slope_het = 0.2,
                 covariate_shift = 0.3, snr = 1.0, seed = seed )
    if variant == 'no_intercept':
        kw['tau'] = 0.0
    if variant == 'intercept_only':
        kw.update( slope_het = 0.0, covariate_shift = 0.0 )
    raw = simulate_dataset( **kw )
    row = { 'tau': tau, 'nonlin': nl, 'n': n, 'rep': rep, 'variant': variant }
    if variant == 'paper':
        X, y, cl, parts = components( **kw )
        assert np.allclose( y, raw.Y[:, 0] ), "generator replica is not exact"
        vy = y.var()
        for k, v in parts.items():
            row[f'tot_{k}'] = v.var() / vy                         # share of the total variance
            row[f'btw_{k}'] = between( v, cl ).var() / vy          # BETWEEN-center share
        row['btw_y'] = between( y, cl ).var() / vy
        # between-center variance NOT explained by the covariates with common slopes: this is what BFI-Hetero needs
        row['btw_resid_common'] = between( y - parts['common_lin'], cl ).var() / vy
    cfg = { 'lambda_local': 1.0, 'lambda_global': 1.0, 'sigma2_mode': 'global',
            'cv': { 'n_splits': 5, 'n_repeats': 1, 'seed': 0 } }
    res = CVRunner( raw, cfg, build_registry( cfg, only = [ 'BFI-Linear', 'BFI-Hetero' ] ) ).run( verbose = False )
    r2  = { m: point_metrics( res.y_true, P )['R2'] for m, P in res.preds.items() }
    ex  = res.extras['BFI-Hetero']
    t2  = np.mean( [ e['tau2'] for e in ex ] ); s2 = np.mean( [ e['sigma2'] for e in ex ] )
    row.update( R2_hetero = r2['BFI-Hetero'], R2_linear = r2['BFI-Linear'],
                dR2_linear = r2['BFI-Linear'] - r2['BFI-Hetero'],
                icc_hat = np.mean( [ e['tau2'] / ( e['tau2'] + e['sigma2'] ) for e in ex ] ),
                tau2_hat = t2, sigma2_hat = s2 )
    return row


if __name__ == '__main__':
    n_rep = int( sys.argv[1] ) if len( sys.argv ) > 1 else 50
    jobs = []
    for tau in ( 0.3, 0.8 ):
        for nl in ( 0.0, 0.3, 0.6 ):
            for n in ( 5, 10, 20, 50, 100 ):
                for r in range( n_rep ):
                    for v in ( 'paper', 'no_intercept', 'intercept_only' ):
                        jobs.append( delayed( one )( tau, nl, n, r, v ) )
    rows = Parallel( n_jobs = -1, verbose = 0 )( jobs )
    df = pd.DataFrame( rows )
    df.to_csv( 'icc_check_raw.csv', index = False )
    print( 'rows', len( df ) )
