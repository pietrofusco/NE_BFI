#!/usr/bin/env python3
"""
Reconstruction risk from the BFI-Hetero message and cost of the mitigations.

A1  "Singleton" attack (no optimization): the message reveals S = W'W and s = W'y.  For a
    binary covariate b (centered with the global mean m, known to the server from the standardization round),
    the row S[b,:] + m * S[0,:] = sum_{i: b_i=1} w_i.  If only one patient in the center has b = 1 (or b = 0,
    by complement), their standardized record (covariates + outcome, from s) is revealed EXACTLY and
    is mapped back to the original scale with the global mean/SD.  The same holds for the products b_j * b_k
    (2x2 table cells equal to 1).
A2  Rotation attack: if n <= rank, D = [1 | X | y] is determined by D'D up to Q in O(n);
    the column of ones fixes one direction, the binary constraints fix the rest.  Optimization over O(n-1).
B   Secure aggregation: the BFI-Hetero server uses center data ONLY through (n_l, 1'X_l, 1'y_l) and the
    SUMS over all centers of X'X, X'y, y'y.  Numerical check that the estimator is identical.
C   R^2 cost of Gaussian noise (DP) on the per-center sums and on the aggregates.
"""
import sys
import numpy  as np
import pandas as pd


from scipy.linalg   import expm
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment
from joblib         import Parallel, delayed
from simulation     import simulate_dataset
from data_handler   import Preprocessor
from data_handler   import make_repeated_cv_splits
from data_handler   import build_fold
from models         import local_bfi_message
from models         import _sufficient_from_message
from models         import aggregate_bfi_hetero
from models         import predict_bfi_hetero
from evaluation     import CVRunner
from evaluation     import build_registry
from evaluation     import point_metrics

P_CONT, P_BIN = 7, 3


# ----------------------------------------------------------------------------- utilities

def center_view( n, seed, n_clusters = 25 ):
    """One dataset simulated as in the paper; global standardization (round 1) known to the server."""
    raw = simulate_dataset( n_clusters = n_clusters, n_per_cluster = n, nonlin = 0.0, tau = 0.8, seed = seed )
    pp  = Preprocessor( raw.groups ).fit( raw.X, raw.Y )
    Xs, ys = pp.transform_X( raw.X ), pp.transform_Y( raw.Y )[:, 0]
    return raw, pp, Xs, ys


def message_stats( Xs, ys ):
    msg = local_bfi_message( Xs, ys, 1.0, 1.0 )
    S, s, yy = _sufficient_from_message( msg )        # what the server recovers from the message
    return S, s, yy


# ----------------------------------------------------------------------------- A1

def singleton_attack( Xs, ys, pp, raw_X, raw_y ):
    """-> number of records reconstructed exactly (original scale) and check."""
    n, p = Xs.shape
    S, s, yy = message_stats( Xs, ys )
    tot_w = S[0, :p + 1]                  # sum_i [1, x_i]
    tot_y = s[0]
    found = {}
    bins  = range( P_CONT, P_CONT + P_BIN )
    for j in bins:
        m   = pp.mean_X[j]                                     # global mean of the binary covariate (known)
        cnt = S[0, 1 + j] + m * n                              # number of b = 1 in the center
        cnt = int( round( cnt ) )
        row_w = S[1 + j, :p + 1] + m * tot_w                   # sum_{b=1} [1, x_i]
        row_y = s[1 + j] + m * tot_y
        for k, wsum, ysum in ( ( 1, row_w, row_y ), ( n - 1, tot_w - row_w, tot_y - row_y ) ):
            if cnt == k:
                x_std = wsum[1:]; y_std = ysum
                x_raw = x_std * pp.std_X + pp.mean_X
                y_raw = pp.inverse_Y( np.array( [[ y_std ]] ) )[0, 0]
                # which patient is it? (check)
                i = int( np.argmin( np.abs( raw_X - x_raw ).sum( 1 ) ) )
                ok = np.allclose( raw_X[i], x_raw, atol = 1e-6 ) and abs( raw_y[i] - y_raw ) < 1e-6
                found[i] = found.get( i, False ) or ok
    return len( found ), all( found.values() ) if found else True


# ----------------------------------------------------------------------------- A2

def _skew( v, k ):
    A = np.zeros( (k, k) ); iu = np.triu_indices( k, 1 ); A[iu] = v; return A - A.T


def rotation_attack( Xs, ys, pp, restarts = 200, rng = None ):
    """Reconstructs D = [1 | X_std | y_std] from D'D (n <= rank).  -> errors with respect to the true data."""
    rng = rng or np.random.default_rng( 0 )
    n, p = Xs.shape
    S, s, yy = message_stats( Xs, ys )
    G = np.zeros( (p + 2, p + 2) )
    G[:p + 1, :p + 1] = S[:p + 1, :p + 1]; G[:p + 1, p + 1] = G[p + 1, :p + 1] = s[:p + 1]; G[p + 1, p + 1] = yy
    ev, V = np.linalg.eigh( G )
    idx = np.argsort( ev )[::-1][:n]
    D0  = ( V[:, idx] * np.sqrt( np.maximum( ev[idx], 0 ) ) ).T          # n x (p+2), D0'D0 = G
    # rotation mapping the first column onto 1_n (Householder)
    a, b = D0[:, 0], np.ones( n )
    v = a - b
    H = np.eye( n ) - 2 * np.outer( v, v ) / ( v @ v ) if np.linalg.norm( v ) > 1e-12 else np.eye( n )
    D1 = H @ D0
    # orthonormal basis of the complement of 1_n
    B  = np.linalg.qr( np.c_[ np.ones( n ), rng.normal( size = (n, n - 1) ) ] )[0][:, 1:]
    bcols = [ 1 + j for j in range( P_CONT, P_CONT + P_BIN ) ]
    lo = np.array( [ -pp.mean_X[j] for j in range( P_CONT, P_CONT + P_BIN ) ] )   # admissible values: lo, lo+1
    k  = n - 1

    def recon( v ):
        R = np.eye( n ) + B @ ( expm( _skew( v, k ) ) - np.eye( k ) ) @ B.T   # rotation that fixes 1_n
        return R @ D1

    def loss( v ):
        D = recon( v )[:, bcols] - lo
        return np.sum( np.minimum( D**2, ( D - 1 )**2 ) )

    best = ( np.inf, None )
    npar = k * ( k - 1 ) // 2
    for _ in range( restarts ):
        r = minimize( loss, rng.uniform( -np.pi, np.pi, npar ), method = 'L-BFGS-B' )
        if r.fun < best[0]:
            best = ( r.fun, r.x )
        if best[0] < 1e-10:
            break
    D = recon( best[1] ) if best[1] is not None else D1
    # reflections are also allowed: try D and the reflection that fixes 1_n
    truth = np.c_[ np.ones( n ), Xs, ys ]
    cands = [ D, ( np.eye( n ) - 2 * np.outer( B[:, 0], B[:, 0] ) ) @ D ]
    out = None
    for Dc in cands:
        C   = ( ( Dc[:, None, 1:] - truth[None, :, 1:] )**2 ).sum( -1 )
        r_, c_ = linear_sum_assignment( C )
        err = np.abs( Dc[r_, 1:] - truth[c_, 1:] )
        bin_ok = np.mean( np.abs( Dc[r_][:, bcols] - truth[c_][:, bcols] ) < 0.5 )
        res = { 'loss': best[0], 'max_abs_err_std': float( err.max() ),
                'median_abs_err_std': float( np.median( err ) ),
                'y_abs_err_std': float( np.abs( Dc[r_, -1] - truth[c_, -1] ).max() ),
                'binary_correct': float( bin_ok ) }
        if out is None or res['max_abs_err_std'] < out['max_abs_err_std']:
            out = res
    return out


# ----------------------------------------------------------------------------- B

def aggregate_secure( pieces, lambda_g = 1.0, lambda_b0 = 1e-6, sigma2 = 1.0, tau2 = 1.0, max_iter = 200,
                      tol = 1e-8, sigma2_bounds = (0.02, 2.0), tau2_bounds = (1e-6, 2.0) ):
    """
    BFI-Hetero server that receives ONLY:
      per center (in the clear): n_l, u_l = [n_l, 1'X_l] (= W_s'1), c_l = 1'y_l
      via secure aggregation   : SS = sum_l W_s'W_s, Sy = sum_l W_s'y, YY = sum_l y'y     (W_s = [1 | X])
    Same EM as models.aggregate_bfi_hetero (Schur + M-step), rewritten in terms of the sums only.
    """
    ids = list( pieces['n'].keys() )
    SS, Sy, YY = pieces['SS'], pieces['Sy'], pieces['YY']
    N = sum( pieces['n'].values() ); d = SS.shape[0]
    prior = np.r_[ lambda_b0, np.full( d - 1, lambda_g ) ]
    it = 0
    for it in range( 1, max_iter + 1 ):
        A_tot = np.diag( prior ) + SS / sigma2; b_tot = Sy / sigma2
        blk = {}
        for c in ids:
            u, nc, yc = pieces['u'][c], pieces['n'][c], pieces['c'][c]
            Asc, Acc, bc = u / sigma2, nc / sigma2 + 1 / tau2, yc / sigma2
            A_tot -= np.outer( Asc, Asc ) / Acc; b_tot -= Asc * bc / Acc
            blk[c] = ( Asc, Acc, bc )
        ev, V = np.linalg.eigh( ( A_tot + A_tot.T ) / 2 )        # robustness with a noisy Gram matrix
        A_tot = ( V * np.maximum( ev, 1.0 ) ) @ V.T
        beta = np.linalg.solve( A_tot, b_tot ); Sig = np.linalg.inv( A_tot )
        alpha, var_a, cov = {}, {}, {}
        for c in ids:
            Asc, Acc, bc = blk[c]
            alpha[c] = ( bc - Asc @ beta ) / Acc
            g = Sig @ Asc / Acc; cov[c] = -g; var_a[c] = 1 / Acc + Asc @ g / Acc
        # M-step: summed E||y - W theta||^2, written with the sums only + per-center terms
        ess = YY - 2 * beta @ Sy + beta @ SS @ beta + np.sum( SS * Sig )
        for c in ids:
            u, nc, yc = pieces['u'][c], pieces['n'][c], pieces['c'][c]
            a = alpha[c]
            ess += -2 * a * yc + 2 * a * ( u @ beta ) + a * a * nc + 2 * u @ cov[c] + nc * var_a[c]
        s2 = float( np.clip( ess / N, *sigma2_bounds ) )
        t2 = float( np.clip( np.mean( [ alpha[c]**2 + var_a[c] for c in ids ] ), *tau2_bounds ) )
        delta = abs( s2 - sigma2 ) / sigma2 + abs( t2 - tau2 ) / tau2
        sigma2, tau2 = s2, t2
        if delta < tol:
            break
    return { 'beta': beta, 'alpha': alpha, 'sigma2': sigma2, 'tau2': tau2, 'n_iter': it }


def secure_pieces( local, eps = None, delta = 1e-5, C = None, rng = None ):
    """
    Local computation + (optional) Gaussian noise.  eps=None: no noise (secure aggregation only).
    With eps: rows [x, y] clipped to L2 norm C; half of the budget to the per-center sums (u, c), half
    to the aggregate (SS, Sy, YY) -- Gaussian mechanism, sensitivity to replacing one record.
    """
    rng = rng or np.random.default_rng( 0 )
    pcs = { 'n': {}, 'u': {}, 'c': {} }
    SS = Sy = YY = 0.0
    for cid, d in local.items():
        X, y = d['X_train'], d['Y_train'].ravel()
        if eps is not None:
            Z = np.c_[ X, y ]; nr = np.linalg.norm( Z, axis = 1, keepdims = True )
            Z = Z * np.minimum( 1.0, C / np.maximum( nr, 1e-12 ) ); X, y = Z[:, :-1], Z[:, -1]
        W = np.c_[ np.ones( len( y ) ), X ]
        pcs['n'][cid] = len( y )
        u, c = W.T @ np.ones( len( y ) ), float( y.sum() )
        # (per-center sums left exact: central DP on the aggregates only, see text)
        pcs['u'][cid], pcs['c'][cid] = u, c
        SS = SS + W.T @ W; Sy = Sy + W.T @ y; YY = YY + y @ y
    if eps is not None:
        sd = 2 * ( C**2 + 1 ) * np.sqrt( 2 * np.log( 1.25 / delta ) ) / eps   # ||ww'||_F <= C^2 + 1
        E = rng.normal( 0, sd, SS.shape ); E = np.triu( E ); E = E + np.triu( E, 1 ).T
        E[0, :] = E[:, 0] = 0                       # intercept row = sum of the u_l, already known
        ey = rng.normal( 0, sd, Sy.shape ); ey[0] = 0
        SS = SS + E; Sy = Sy + ey; YY = YY + rng.normal( 0, sd )
        # PSD projection of the noisy Gram matrix
        ev, V = np.linalg.eigh( SS ); SS = ( V * np.maximum( ev, 1e-6 ) ) @ V.T
    pcs.update( SS = SS, Sy = Sy, YY = float( YY ) )
    return pcs


def fit_predict_secure( fold, cfg, seed, eps = None, C = None ):
    pcs = secure_pieces( fold['local'], eps = eps, C = C, rng = np.random.default_rng( seed ) )
    out = aggregate_secure( pcs, lambda_g = cfg.get( 'lambda_global', 1.0 ) )
    te  = fold['pooled_test']
    return { 'pred': predict_bfi_hetero( out, te['X'], te['cluster'] ), 'extras': {} }


def check_exactness():
    worst = 0.0
    for seed in range( 10 ):
        raw = simulate_dataset( n_clusters = 25, n_per_cluster = 5 + 10 * ( seed % 3 ), tau = 0.8, seed = seed )
        s   = make_repeated_cv_splits( raw.clusters, 5, 1, seed = 0 )[0]
        f   = build_fold( raw, s['train_idx'], s['test_idx'] )
        feats = { c: d['X_train'] for c, d in f['local'].items() }; Ys = { c: d['Y_train'] for c, d in f['local'].items() }
        ref = aggregate_bfi_hetero( { c: local_bfi_message( feats[c], Ys[c] ) for c in feats }, lambda_g = 1.0 )
        sec = aggregate_secure( secure_pieces( f['local'] ), lambda_g = 1.0 )
        te  = f['pooled_test']
        d1  = np.max( np.abs( predict_bfi_hetero( ref, te['X'], te['cluster'] ) - predict_bfi_hetero( sec, te['X'], te['cluster'] ) ) )
        worst = max( worst, d1, abs( ref['tau2'] - sec['tau2'] ), abs( ref['sigma2'] - sec['sigma2'] ) )
    return worst


# ----------------------------------------------------------------------------- C

def dp_one( tau, n, rep, eps_list, C ):
    raw = simulate_dataset( n_clusters = 25, n_per_cluster = n, nonlin = 0.0, tau = tau, seed = 50_000 + rep )
    cfg = { 'lambda_local': 1.0, 'lambda_global': 1.0, 'sigma2_mode': 'global', 'cv': { 'n_splits': 5, 'n_repeats': 1, 'seed': 0 } }
    reg = build_registry( cfg, only = [ 'BFI-Linear', 'BFI-Hetero' ] )
    reg['SecAgg (exact)'] = lambda fold, seed: fit_predict_secure( fold, cfg, seed )
    for e in eps_list:
        reg[f'SecAgg+DP eps={e:g}'] = ( lambda fold, seed, e = e: fit_predict_secure( fold, cfg, seed, eps = e, C = C ) )
    res = CVRunner( raw, cfg, reg ).run( verbose = False )
    r2  = { m: point_metrics( res.y_true, P )['R2'] for m, P in res.preds.items() }
    return [ { 'tau': tau, 'n': n, 'rep': rep, 'model': m, 'dR2': v - r2['BFI-Hetero'], 'R2': v } for m, v in r2.items() ]


if __name__ == '__main__':
    what = sys.argv[1] if len( sys.argv ) > 1 else 'all'

    if what in ( 'all', 'A' ):
        rows = []
        for n in ( 5, 10, 20, 40 ):
            for seed in range( 40 ):
                raw, pp, Xs, ys = center_view( n, 1000 + seed )
                for c in np.unique( raw.clusters ):
                    m = raw.clusters == c
                    k, ok = singleton_attack( Xs[m], ys[m], pp, raw.X[m], raw.Y[m, 0] )
                    rows.append( { 'n_target': n, 'n_center': int( m.sum() ), 'records_exposed': k, 'verified': ok } )
        A1 = pd.DataFrame( rows )
        A1.to_csv( 'privacy_singleton.csv', index = False )
        print( A1.groupby( 'n_target' ).agg( centers = ( 'records_exposed', 'size' ),
                                             pct_centers_exposed = ( 'records_exposed', lambda v: 100 * ( v > 0 ).mean() ),
                                             mean_records = ( 'records_exposed', 'mean' ),
                                             pct_patients = ( 'records_exposed', 'sum' ),
                                             all_verified = ( 'verified', 'all' ) ).assign(
                pct_patients = lambda d: 100 * d.pct_patients / A1.groupby( 'n_target' )['n_center'].sum() ).round( 2 ) )

        rows = []
        for n in ( 4, 5, 6, 8 ):
            for t in range( 10 ):
                raw, pp, Xs, ys = center_view( max( n, 5 ), 2000 + t )
                m = np.where( raw.clusters == 0 )[0][:n]
                r = rotation_attack( Xs[m], ys[m], pp, restarts = 300, rng = np.random.default_rng( t ) )
                r.update( n = n, trial = t ); rows.append( r )
        A2 = pd.DataFrame( rows ); A2.to_csv( 'privacy_rotation.csv', index = False )
        A2['exact'] = A2['max_abs_err_std'] < 1e-4
        print( A2.groupby( 'n' ).agg( exact_recon_pct = ( 'exact', lambda v: 100 * v.mean() ),
                                      median_err = ( 'median_abs_err_std', 'median' ),
                                      binary_ok = ( 'binary_correct', 'mean' ) ).round( 4 ) )

    if what in ( 'all', 'B' ):
        print( 'max |BFI-Hetero - SecAgg| (pred, tau2, sigma2):', check_exactness() )

    if what in ( 'all', 'C' ):
        n_rep = int( sys.argv[2] ) if len( sys.argv ) > 2 else 20
        eps_list = ( 1, 3, 10, 30 )
        C = np.sqrt( 2 * 14 )                      # ~1.4 x typical norm of standardized [x, y] (p = 13)
        jobs = [ delayed( dp_one )( tau, n, r, eps_list, C ) for tau in ( 0.3, 0.8 ) for n in ( 5, 10, 20, 50 )
                 for r in range( n_rep ) ]
        out = Parallel( n_jobs = -1 )( jobs )
        D = pd.DataFrame( [ r for rr in out for r in rr ] ); D.to_csv( 'privacy_dp.csv', index = False )
        g = D.groupby( [ 'tau', 'n', 'model' ] )['dR2']
        print( 'MEDIAN dR2'); print( g.median().unstack( 'n' ).round( 3 ).to_string() )
        print( 'share of fits with R2 < 0' ); print( D.assign( neg = D.R2 < 0 ).groupby( [ 'tau', 'n', 'model' ] )['neg'].mean().unstack( 'n' ).round( 2 ).to_string() )
