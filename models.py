#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py - all models used in the study.

Contents
  1. LocalBayesianGLM + aggregate_bfi + fit_bfi            (BFI, Jonker et al. 2024)
  2. Heterogeneous BFI (global intercept + local center intercepts, sigma^2 and tau^2 by EM)
  3. Comparators: BFI-Fixed (Jonker et al. 2024), DLMM (Luo et al. 2022), iterative federated EM-LMM
  4. NeuralFeatureExtractor + FederatedFeatureExtractorTrainer (FedAvg/FedProx) + gradient_importance
  5. fit + predict functions on a single CV fold
  6. BaselineModels (local and pooled Ridge/MLP, random-intercept LMM)

BFI aggregation (global prior counted once):
    A_global     = sum_l ( A_l - lambda_l I ) + lambda_g I
    theta_global = A_global^{-1} sum_l A_l theta_l
For a Gaussian linear model with common sigma^2 this is exactly pooled ridge
(alpha = sigma^2 * lambda_g).

torch and statsmodels are imported lazily, only by the models that need them.
All fit+predict functions work on the standardized scale of the fold and return
{'pred': (N_test,), 'extras': {...}}, aligned with fold['pooled_test'].
"""
# %%

import copy
import warnings
import numpy    as np


from sklearn.exceptions     import ConvergenceWarning
from sklearn.linear_model   import Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor

try:
    import torch
except Exception:                                    # neural part unavailable (torch missing or broken)
    torch = None


# %% ========================================================== 1. BFI

class LocalBayesianGLM:
    """
    y | theta ~ N( Z theta, sigma^2 I ),   theta ~ N( 0, (1/lambda_reg) I )
        theta_hat = ( Z'Z + sigma^2 lambda_reg I )^{-1} Z'Y
        A_hat     = kron( I_q, Z'Z / sigma^2 + lambda_reg I )
    """

    def __init__( self,
                  Y,
                  Z,
                  lambda_reg = 1.0,
                  sigma2     = 1.0 ):

        self.Y            = Y if Y.ndim > 1 else Y.reshape(-1, 1)
        self.Z            = Z
        self.lambda_reg   = lambda_reg
        self.sigma2       = sigma2
        self.n, self.k    = Z.shape
        self.num_outcomes = self.Y.shape[1]
        self.dim_theta    = self.k * self.num_outcomes
        self._ZtZ         = Z.T @ Z
        self._ZtY         = Z.T @ self.Y


    def _negative_log_posterior( self,
                                 theta_flat ):
        """Only used as a numerical check in the tests."""

        theta = theta_flat.reshape( self.num_outcomes, self.k )
        mu    = self.Z @ theta.T

        return 0.5 * np.sum( (self.Y - mu)**2 ) / self.sigma2 + 0.5 * self.lambda_reg * np.sum( theta**2 )


    def fit( self ):

        eye            = np.eye( self.k )
        theta_mat      = np.linalg.solve( self._ZtZ + self.sigma2 * self.lambda_reg * eye, self._ZtY )
        self.theta_hat = theta_mat.T.flatten()
        A_block        = self._ZtZ / self.sigma2 + self.lambda_reg * eye
        self.A_hat     = np.kron( np.eye(self.num_outcomes), A_block )

        return self.theta_hat, self.A_hat


    def residual_ss( self,
                     theta = None ):
        """Residual sum of squares (a scalar that can be shared with the server)."""

        theta = self.theta_hat if theta is None else theta
        mu    = self.Z @ theta.reshape( self.num_outcomes, self.k ).T

        return float( np.sum( (self.Y - mu)**2 ) )


    def effective_df( self ):
        """Effective degrees of freedom of the local ridge: tr(H) per output."""

        H = np.linalg.solve( self._ZtZ + self.sigma2 * self.lambda_reg * np.eye(self.k), self._ZtZ )

        return float( np.trace(H) * self.num_outcomes )


def aggregate_bfi( local_results,
                   lambda_global = 1.0 ):
    """-> (theta_global, A_global). local_results: list of {'theta_hat','A_hat','lambda_reg'}."""

    d     = local_results[0]['theta_hat'].shape[0]
    eye   = np.eye( d )
    A     = lambda_global * eye                                  # global prior, counted once
    A_th  = np.zeros( d )
    for r in local_results:

        A    = A + r['A_hat'] - r['lambda_reg'] * eye            # local likelihood
        A_th = A_th + r['A_hat'] @ r['theta_hat']

    return np.linalg.solve( A, A_th ), A


def _fit_and_aggregate( feats,
                        Ys,
                        lambda_local,
                        lambda_global,
                        sigma2_by_cluster ):

    models, results = {}, []
    for c in feats:

        m                = LocalBayesianGLM( Y          = Ys[c],
                                             Z          = feats[c],
                                             lambda_reg = lambda_local,
                                             sigma2     = sigma2_by_cluster[c] )
        theta_hat, A_hat = m.fit()
        models[c]        = m
        results.append( { 'cluster_id': c, 'theta_hat': theta_hat,
                          'A_hat': A_hat, 'lambda_reg': lambda_local } )

    theta, A = aggregate_bfi( results, lambda_global )

    return models, results, theta, A


def fit_bfi( feats,
             Ys,
             lambda_local  = 1.0,
             lambda_global = 1.0,
             sigma2_mode   = 'global',
             sigma2_fixed  = 1.0,
             sigma2_bounds = (0.02, 2.0) ):
    """
    feats, Ys: {cluster: (n_c, k)} / {cluster: (n_c, q)}.
    sigma2_mode: 'fixed' | 'local' (RSS_l/(n_l-df_l)) | 'global' (two passes, one scalar per center).
    Returns {'theta' (flat), 'A_global', 'sigma2' (dict), 'local_results'}.
    """
    if sigma2_mode not in ( 'fixed', 'local', 'global' ):
        raise ValueError( f"Unknown sigma2_mode: {sigma2_mode}" )

    lo, hi = sigma2_bounds
    s2     = { c: sigma2_fixed for c in feats }
    models, results, theta, A = _fit_and_aggregate( feats, Ys, lambda_local, lambda_global, s2 )

    if sigma2_mode == 'local':

        s2 = { c: float( np.clip( m.residual_ss() / max( m.n * m.num_outcomes - m.effective_df(), 1.0 ), lo, hi ) )
               for c, m in models.items() }

    elif sigma2_mode == 'global':

        rss    = sum( m.residual_ss( theta ) for m in models.values() )
        n_res  = sum( m.n * m.num_outcomes   for m in models.values() )
        p_eff  = A.shape[0] - lambda_global * np.trace( np.linalg.inv(A) )
        s2_hat = float( np.clip( rss / max( n_res - p_eff, 1.0 ), lo, hi ) )
        s2     = { c: s2_hat for c in feats }

    if sigma2_mode != 'fixed':
        models, results, theta, A = _fit_and_aggregate( feats, Ys, lambda_local, lambda_global, s2 )

    return { 'theta': theta, 'A_global': A, 'sigma2': s2, 'local_results': results }


# %% ========================================================== 2. Heterogeneous BFI
#
#   Model:  y_ci = 1 * beta_0 + x_ci' beta + alpha_c + eps_ci,   eps ~ N(0, sigma^2)
#           beta ~ N(0, I/lambda_g)  (global prior, counted once)
#           alpha_c ~ N(0, tau^2)    (center intercept: local parameter, never aggregated)
#
#   Each center sends the BFI message (theta_hat_l, A_l, rss_l, n_l) once, computed with known
#   priors (lambda_s on beta, lambda_c on alpha_c, sigma^2 = 1).  Since the priors are known, the
#   server recovers the sufficient statistics S_l = W_l'W_l, s_l = W_l'y_l, yy_l = y_l'y_l
#   (W_l = [1 | X_l | 1]) exactly, and can therefore:
#     (i)  marginalize alpha_c (Schur complement) and solve for beta;
#     (ii) estimate sigma^2 and tau^2 by EM with no further communication.  Because beta has an
#          (almost) flat prior and is integrated out in the E step, EM maximizes the restricted
#          likelihood: as lambda_g -> 0 the result matches a REML LMM fitted on the pooled data
#          (checked in test_pipeline.py).
#   Note: sharing (S_l, s_l, yy_l) amounts to sharing the center's sufficient statistics, so the
#   disclosure risk for small n_l should be discussed.

def local_bfi_message( X,
                       y,
                       lambda_s = 1.0,
                       lambda_c = 1.0 ):
    """Local computation (inside the center): one-shot BFI message."""

    y   = np.asarray( y, dtype = float ).ravel()
    W   = np.hstack( [ np.ones( (len(y), 1) ), X, np.ones( (len(y), 1) ) ] )   # [b0 | beta | alpha_c]
    d   = W.shape[1]
    P   = np.diag( np.r_[ np.full( d - 1, lambda_s ), lambda_c ] )
    A   = W.T @ W + P                                                         # sigma^2 = 1
    th  = np.linalg.solve( A, W.T @ y )
    rss = float( np.sum( (y - W @ th)**2 ) )

    return { 'theta_hat': th, 'A_hat': A, 'rss': rss, 'n': int(len(y)),
             'lambda_s': lambda_s, 'lambda_c': lambda_c }


def _sufficient_from_message( msg ):
    """Server side: (theta_hat, A, rss) + known priors  ->  (W'W, W'y, y'y)."""

    d  = msg['A_hat'].shape[0]
    P  = np.diag( np.r_[ np.full( d - 1, msg['lambda_s'] ), msg['lambda_c'] ] )
    S  = msg['A_hat'] - P
    s  = msg['A_hat'] @ msg['theta_hat']
    th = msg['theta_hat']
    yy = msg['rss'] + 2.0 * th @ s - th @ S @ th

    return S, s, float(yy)


def aggregate_bfi_hetero( messages,
                          lambda_g       = 1.0,
                          lambda_b0      = 1e-6,
                          sigma2         = 1.0,
                          tau2           = 1.0,
                          estimate       = True,
                          max_iter       = 200,
                          tol            = 1e-8,
                          sigma2_bounds  = (1e-3, 1e3),
                          tau2_bounds    = (1e-6, 1e3) ):
    """
    Server side.  messages: {cluster: output of local_bfi_message}.
    estimate=True: sigma^2 and tau^2 estimated by EM (REML) on the server; False: kept fixed.
    Returns {'beta' (global intercept at index 0), 'alpha' {c: float},
             'sigma2', 'tau2', 'A_global', 'n_iter'}.
    """
    ids   = list( messages.keys() )
    suff  = { c: _sufficient_from_message( messages[c] ) for c in ids }
    N     = sum( messages[c]['n'] for c in ids )
    d     = suff[ids[0]][0].shape[0] - 1                       # size of [b0 | beta]
    prior = np.r_[ lambda_b0, np.full( d - 1, lambda_g ) ]

    it = 0
    for it in range( 1, max_iter + 1 ):

        A_tot, b_tot, blocks = np.diag( prior ), np.zeros( d ), {}
        for c in ids:

            S, s, _ = suff[c]
            Ass, Asc, Acc = S[:d, :d] / sigma2, S[:d, d] / sigma2, S[d, d] / sigma2 + 1.0 / tau2
            bs,  bc       = s[:d] / sigma2, s[d] / sigma2
            A_tot += Ass - np.outer( Asc, Asc ) / Acc
            b_tot += bs - Asc * bc / Acc
            blocks[c] = ( Asc, Acc, bc )

        beta    = np.linalg.solve( A_tot, b_tot )
        Sig_b   = np.linalg.inv( A_tot )
        alpha, var_a, cov_ba = {}, {}, {}
        for c in ids:

            Asc, Acc, bc = blocks[c]
            alpha[c]  = ( bc - Asc @ beta ) / Acc
            g         = Sig_b @ Asc / Acc
            cov_ba[c] = -g                                       # Cov(beta, alpha_c)
            var_a[c]  = 1.0 / Acc + Asc @ g / Acc

        if not estimate:
            break

        # ---- M step (EM -> REML): expectations under the joint posterior of (beta, alpha)
        ess = 0.0
        for c in ids:

            S, s, yy = suff[c]
            th       = np.r_[ beta, alpha[c] ]
            C        = np.zeros( (d + 1, d + 1) )
            C[:d, :d], C[:d, d], C[d, :d], C[d, d] = Sig_b, cov_ba[c], cov_ba[c], var_a[c]
            ess     += yy - 2.0 * th @ s + th @ S @ th + np.sum( S * C )

        s2_new = float( np.clip( ess / N, *sigma2_bounds ) )
        t2_new = float( np.clip( np.mean( [ alpha[c]**2 + var_a[c] for c in ids ] ), *tau2_bounds ) )
        delta  = abs( s2_new - sigma2 ) / sigma2 + abs( t2_new - tau2 ) / tau2
        sigma2, tau2 = s2_new, t2_new
        if delta < tol:
            break

    return { 'beta': beta, 'alpha': alpha, 'sigma2': sigma2, 'tau2': tau2,
             'A_global': A_tot, 'n_iter': it }


def fit_bfi_hetero( feats,
                    Ys,
                    lambda_g      = 1.0,
                    lambda_s      = 1.0,
                    lambda_c      = 1.0,
                    estimate      = True,
                    sigma2_bounds = (1e-3, 1e3),
                    max_iter      = 200,
                    tol           = 1e-8 ):
    """Local messages (one-shot) + aggregation on the server.
    max_iter/tol control the EM iterations, which run on the server and cost no communication."""

    msgs = { c: local_bfi_message( feats[c], Ys[c], lambda_s, lambda_c ) for c in feats }

    return aggregate_bfi_hetero( msgs, lambda_g = lambda_g, estimate = estimate,
                                 sigma2 = 1.0, tau2 = 1.0 / lambda_c,
                                 sigma2_bounds = sigma2_bounds, max_iter = max_iter, tol = tol )


def predict_bfi_hetero( out,
                        X,
                        clusters ):
    """Seen centers: x'beta + alpha_c.  New centers: alpha = 0 (population mean)."""

    pred = out['beta'][0] + X @ out['beta'][1:]
    a    = np.array( [ out['alpha'].get( c, 0.0 ) for c in clusters ] )

    return pred + a


# %% ========================================================== 3. Comparators
#
#   (a) BFI-Fixed  -- BFI for heterogeneous populations with fixed center intercepts
#       (Jonker et al. 2024, "Bayesian Federated Inference for regression models based on non-shared
#       multicenter data sets from heterogeneous populations").  Aggregated model
#           y_ci = gamma_c + x_ci' beta + eps_ci,   eps ~ N(0, sigma^2)
#       with no overall intercept; independent zero-mean Gaussian priors with known precision
#       (beta: lambda_g; gamma_c: lambda_int, 0.001 or 0.1 in the paper).  No between-center variance
#       is estimated: gamma_c is estimated from the patients of center c only (shrinkage towards 0 set
#       by the prior, not learned from the data).  One round: each center sends (theta_hat_c, A_c) of
#       its local model [beta | gamma_c]; the server applies the BFI formula with the center-specific
#       parameters embedded in the global vector (Appendix II.B of Jonker et al.):
#           A_BFI     = sum_c E_c ( A_c - Lambda_c ) E_c' + Lambda_g
#           theta_BFI = A_BFI^{-1} sum_c E_c A_c theta_c
#       where E_c maps [beta | gamma_c] into [beta | gamma_1 ... gamma_L].  The common sigma^2 is
#       estimated on the server from the scalar rss_c in the message (no extra round).
#
#   (b) DLMM  -- Luo et al. 2022 (Nat Commun 13:1678), "DLMM as a lossless one-shot algorithm for
#       collaborative multi-site distributed linear mixed models".  Each center sends the aggregate
#       statistics (X~'X~, X~'y, y'y, n) once, with X~ = [1 | X]; the server maximizes the likelihood
#       (REML by default, ML optional) of the random-intercept model and computes the BLUPs.  For a
#       random intercept  V_c = sigma^2 (I + g 1 1'),  g = tau^2 / sigma^2,
#           V_c^{-1} = ( I - w_c 1 1' ) / sigma^2,   w_c = g / (1 + n_c g),   log|V_c| = n_c log sigma^2 + log(1 + n_c g)
#       so everything depends on the aggregate statistics only: sigma^2 is profiled out analytically
#       and what remains is a scalar optimization over log g.  DLMM is also one-shot and lossless;
#       numerically it matches the pooled REML LMM and BFI-Hetero as lambda_g -> 0.  This is an
#       independent implementation (direct maximization, not EM), so it also serves as a check.
#
#   (c) Federated EM-LMM (iterative) -- reference for the communication cost: the same EM as
#       BFI-Hetero run as a standard distributed algorithm, where each iteration takes one round
#       (the server sends the current (sigma^2, tau^2), the centers return their E-step contributions).
#       With a budget of R rounds, EM may stop before convergence.  Emulated in-process: the per-
#       iteration estimates are identical to the E/M steps of aggregate_bfi_hetero.

def local_bfi_fixed_message( X,
                             y,
                             lambda_s   = 1.0,
                             lambda_int = 1e-3 ):
    """Local computation for BFI-Fixed: MAP and curvature of the model [beta | gamma_c], sigma^2 = 1."""

    y   = np.asarray( y, dtype = float ).ravel()
    W   = np.hstack( [ X, np.ones( (len(y), 1) ) ] )                              # [beta | gamma_c]
    d   = W.shape[1]
    P   = np.diag( np.r_[ np.full( d - 1, lambda_s ), lambda_int ] )
    A   = W.T @ W + P
    th  = np.linalg.solve( A, W.T @ y )
    rss = float( np.sum( (y - W @ th)**2 ) )

    return { 'theta_hat': th, 'A_hat': A, 'rss': rss, 'n': int(len(y)),
             'lambda_s': lambda_s, 'lambda_c': lambda_int }


def aggregate_bfi_fixed( messages,
                         lambda_g        = 1.0,
                         lambda_int      = 1e-3,
                         sigma2          = 1.0,
                         estimate_sigma2 = True,
                         max_iter        = 100,
                         tol             = 1e-10,
                         sigma2_bounds   = (1e-3, 1e3) ):
    """
    BFI-Fixed server (BFI formula with center-specific parameters).
    messages: {cluster: output of local_bfi_fixed_message}.
    estimate_sigma2: sigma^2 = RSS(theta_BFI) / (N - p_eff) as a fixed point (sigma^2 only enters
    through its ratio with the priors); the RSS terms are functions of the message (theta_c, A_c, rss_c).
    -> {'beta', 'gamma' {c: float}, 'sigma2', 'A_global', 'n_iter'}
    """
    ids  = list( messages.keys() )
    L    = len( ids )
    suff = { c: _sufficient_from_message( messages[c] ) for c in ids }      # (W'W, W'y, y'y), W = [X | 1]
    N    = sum( messages[c]['n'] for c in ids )
    p    = suff[ids[0]][0].shape[0] - 1
    D    = p + L
    Lam  = np.r_[ np.full( p, lambda_g ), np.full( L, lambda_int ) ]
    emb  = { c: np.r_[ np.arange( p ), p + j ] for j, c in enumerate( ids ) }   # E_c as indices

    it = 0
    for it in range( 1, max_iter + 1 ):

        A, b = np.diag( Lam ), np.zeros( D )
        for c in ids:

            S, s, _ = suff[c]
            ix      = emb[c]
            A[np.ix_( ix, ix )] += S / sigma2                   # E_c (A_c - Lambda_c) E_c' / sigma^2
            b[ix]               += s / sigma2                   # E_c A_c theta_c / sigma^2
        theta = np.linalg.solve( A, b )

        if not estimate_sigma2:
            break

        rss = 0.0
        for c in ids:
            S, s, yy = suff[c]
            th_c     = theta[emb[c]]
            rss     += yy - 2.0 * th_c @ s + th_c @ S @ th_c
        p_eff  = D - float( np.sum( np.diag( np.linalg.inv( A ) ) * Lam ) )
        s2_new = float( np.clip( rss / max( N - p_eff, 1.0 ), *sigma2_bounds ) )
        delta  = abs( s2_new - sigma2 ) / sigma2
        sigma2 = s2_new
        if delta < tol:
            break

    return { 'beta': theta[:p], 'gamma': { c: float( theta[p + j] ) for j, c in enumerate( ids ) },
             'sigma2': float( sigma2 ), 'A_global': A, 'n_iter': it }


def predict_bfi_fixed( out,
                       X,
                       clusters ):
    """x'beta + gamma_c.  Unseen centers: unweighted mean of the intercepts (outside the scope of the study)."""

    g0 = float( np.mean( list( out['gamma'].values() ) ) )
    g  = np.array( [ out['gamma'].get( c, g0 ) for c in clusters ] )

    return X @ out['beta'] + g


def local_dlmm_summary( X,
                        y ):
    """Local computation for DLMM: aggregate statistics, sent once."""

    y  = np.asarray( y, dtype = float ).ravel()
    Xt = np.hstack( [ np.ones( (len(y), 1) ), X ] )

    return { 'XtX': Xt.T @ Xt, 'Xty': Xt.T @ y, 'yty': float( y @ y ), 'n': int(len(y)) }


def _dlmm_pieces( summ,
                  g ):
    """GLS given g = tau^2/sigma^2: M = X'V^{-1}X sigma^2, m = X'V^{-1}y sigma^2, r = (y-Xb)'V^{-1}(y-Xb) sigma^2."""

    k = summ[0]['XtX'].shape[0]
    M, m, q, logdet = np.zeros( (k, k) ), np.zeros( k ), 0.0, 0.0
    for sc in summ:
        n, u, s1 = sc['n'], sc['XtX'][0], sc['Xty'][0]               # u = X~'1 (first row), s1 = 1'y
        w        = g / ( 1.0 + n * g )
        M       += sc['XtX'] - w * np.outer( u, u )
        m       += sc['Xty'] - w * u * s1
        q       += sc['yty'] - w * s1**2
        logdet  += np.log1p( n * g )
    beta = np.linalg.solve( M, m )
    r    = float( q - m @ beta )

    return beta, r, M, logdet


def fit_dlmm( summaries,
              reml             = True,
              log_g_bounds     = (-20.0, 10.0) ):
    """
    DLMM server.  summaries: {cluster: local_dlmm_summary}.
    Profile of -2 log L (sigma^2 profiled out):
        REML:  (N - k) log( r / (N - k) ) + sum_c log(1 + n_c g) + log|M|
        ML  :   N      log( r /  N      ) + sum_c log(1 + n_c g)
    optimized over t = log g (bounded scalar search); the boundary g -> 0 lies inside the interval.
    -> {'beta' (intercept at index 0), 'alpha' {c: BLUP}, 'sigma2', 'tau2', 'n_eval'}
    """
    from scipy.optimize import minimize_scalar

    ids  = list( summaries.keys() )
    summ = [ summaries[c] for c in ids ]
    N    = sum( sc['n'] for sc in summ )
    k    = summ[0]['XtX'].shape[0]
    dof  = N - k if reml else N

    def obj( t ):
        beta, r, M, logdet = _dlmm_pieces( summ, np.exp( t ) )
        val = dof * np.log( max( r, 1e-300 ) / dof ) + logdet
        if reml:
            val += np.linalg.slogdet( M )[1]
        return val

    res  = minimize_scalar( obj, bounds = log_g_bounds, method = 'bounded',
                            options = { 'xatol': 1e-10, 'maxiter': 500 } )
    g    = float( np.exp( res.x ) )
    beta, r, _, _ = _dlmm_pieces( summ, g )
    sigma2 = r / dof
    alpha  = {}
    for c, sc in zip( ids, summ ):
        w        = g / ( 1.0 + sc['n'] * g )
        alpha[c] = float( w * ( sc['Xty'][0] - sc['XtX'][0] @ beta ) )   # BLUP: tau^2 1'V^{-1}(y - X b)

    return { 'beta': beta, 'alpha': alpha, 'sigma2': float( sigma2 ), 'tau2': float( g * sigma2 ),
             'n_eval': int( res.nfev ) }


def message_floats( model,
                    p ):
    """Number of values sent by one center in one round (symmetric matrices: upper triangle only)."""

    sym = lambda d: d * ( d + 1 ) // 2
    if model == 'BFI-Linear':                         # theta, A, + y'y (global sigma^2 without an extra round)
        return p + sym( p ) + 2
    if model == 'BFI-Fixed':                          # [beta | gamma_c]: theta, A, rss, n
        return ( p + 1 ) + sym( p + 1 ) + 2
    if model in ( 'BFI-Hetero', 'Federated EM-LMM' ):  # [b0 | beta | alpha_c]: theta, A, rss, n
        return ( p + 2 ) + sym( p + 2 ) + 2
    if model == 'DLMM':                               # X~'X~, X~'y, y'y, n
        return sym( p + 1 ) + ( p + 1 ) + 2
    raise ValueError( model )


# %% ========================================================== 4. Neural part (torch)

if torch is not None:

    class NeuralFeatureExtractor( torch.nn.Module ):
        """Autoencoder + supervised linear head Z->Y (sharing the encoder)."""

        def __init__( self,
                      input_dim,
                      latent_dim = 8,
                      hidden_dim = 16,
                      output_dim = 1 ):

            super().__init__()
            self.encoder   = torch.nn.Sequential( torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
                                                  torch.nn.Linear(hidden_dim, latent_dim) )
            self.decoder   = torch.nn.Sequential( torch.nn.Linear(latent_dim, hidden_dim), torch.nn.ReLU(),
                                                  torch.nn.Linear(hidden_dim, input_dim) )
            self.predictor = torch.nn.Linear( latent_dim, output_dim )


        def forward( self, x ):

            z = self.encoder( x )

            return z, self.decoder(z), self.predictor(z)


        def train_model( self,
                         X,
                         Y            = None,
                         epochs       = 300,
                         lr           = 1e-3,
                         alpha_recon  = 0.1,
                         beta_pred    = 1.0,
                         global_state = None,
                         mu_prox      = 0.0 ):
            """
            loss = alpha_recon * MSE(X_hat, X) + beta_pred * MSE(Y_hat, Y)
                   [+ mu_prox/2 * ||w - w_global||^2  (FedProx, if global_state is given and mu_prox > 0)]
            """
            opt      = torch.optim.Adam( self.parameters(), lr = lr )
            mse      = torch.nn.MSELoss()
            X_t      = torch.FloatTensor( X )
            Y_t      = torch.FloatTensor( Y ) if Y is not None else None
            g_params = None
            if global_state is not None and mu_prox > 0:
                g_params = { k: v.clone().detach() for k, v in global_state.items() }

            self.train()
            for _ in range( epochs ):

                opt.zero_grad()
                _, x_rec, y_hat = self.forward( X_t )
                loss = alpha_recon * mse( x_rec, X_t )
                if Y_t is not None:
                    loss = loss + beta_pred * mse( y_hat, Y_t )
                if g_params is not None:
                    loss = loss + 0.5 * mu_prox * sum( torch.sum( (p - g_params[n])**2 )
                                                       for n, p in self.named_parameters() )
                loss.backward()
                opt.step()


        def loss_value( self,
                        X,
                        Y,
                        alpha_recon = 0.1,
                        beta_pred   = 1.0 ):
            """Training objective (without the proximal term) evaluated on (X, Y); used for early stopping."""

            self.eval()
            mse = torch.nn.MSELoss()
            with torch.no_grad():
                X_t = torch.FloatTensor( X )
                _, x_rec, y_hat = self.forward( X_t )
                loss = alpha_recon * mse( x_rec, X_t )
                if beta_pred > 0 and Y is not None:
                    loss = loss + beta_pred * mse( y_hat, torch.FloatTensor( Y ) )

            return float( loss )


        def fit_early_stopping( self,
                                X,
                                Y,
                                X_val,
                                Y_val,
                                max_epochs  = 300,
                                check_every = 15,
                                patience    = 3,
                                lr          = 1e-3,
                                alpha_recon = 0.1,
                                beta_pred   = 1.0 ):
            """
            Centralized full-batch training with a single Adam optimizer.  Every `check_every` epochs the
            validation loss is evaluated and the best weights are kept (epoch 0 included); training stops
            after `patience` checks without improvement.  Same checking scheme as the federated trainer
            (check_every = local_epochs, max_epochs = rounds * local_epochs), so the budgets are comparable.
            """
            opt  = torch.optim.Adam( self.parameters(), lr = lr )
            mse  = torch.nn.MSELoss()
            X_t, Y_t = torch.FloatTensor( X ), torch.FloatTensor( Y )

            best_v, best_ep, bad = self.loss_value( X_val, Y_val, alpha_recon, beta_pred ), 0, 0
            best_state = copy.deepcopy( self.state_dict() )
            ep = 0
            for ep in range( 1, max_epochs + 1 ):

                self.train()
                opt.zero_grad()
                _, x_rec, y_hat = self.forward( X_t )
                loss = alpha_recon * mse( x_rec, X_t )
                if beta_pred > 0:
                    loss = loss + beta_pred * mse( y_hat, Y_t )
                loss.backward()
                opt.step()

                if ep % check_every == 0:

                    v = self.loss_value( X_val, Y_val, alpha_recon, beta_pred )
                    if v < best_v - 1e-10:
                        best_v, best_ep, bad = v, ep, 0
                        best_state = copy.deepcopy( self.state_dict() )
                    else:
                        bad += 1
                        if bad >= patience:
                            break

            self.load_state_dict( best_state )

            return { 'best_epoch': best_ep, 'epochs_run': ep, 'val_loss': best_v }


        def extract_features( self, X ):

            self.eval()
            with torch.no_grad():
                Z, _, _ = self.forward( torch.FloatTensor( X ) )

            return Z.numpy()


        def get_flat_state( self ):

            return copy.deepcopy( self.state_dict() )


        def load_flat_state( self, state_dict ):

            self.load_state_dict( state_dict )


class FederatedFeatureExtractorTrainer:
    """
    FedAvg / FedProx (McMahan 2017; Li 2020): in each round every center trains for `local_epochs`
    epochs starting from the global weights; the server averages the weights with weight n_c / n_tot.
    In-process simulation: network communication is not modelled.
    """

    def __init__( self,
                  model_factory,
                  rounds       = 20,
                  local_epochs = 5,
                  lr           = 1e-3,
                  alpha_recon  = 0.1,
                  beta_pred    = 1.0,
                  mu_prox      = 0.0 ):

        self.model_factory = model_factory
        self.rounds, self.local_epochs, self.lr = rounds, local_epochs, lr
        self.alpha_recon, self.beta_pred, self.mu_prox = alpha_recon, beta_pred, mu_prox
        self.global_model  = model_factory()


    @staticmethod
    def _weighted_average( state_dicts,
                           weights ):

        w   = np.asarray( weights, dtype = float ) / float( np.sum(weights) )
        avg = copy.deepcopy( state_dicts[0] )
        for key in avg:
            avg[key] = sum( sd[key].float() * wi for sd, wi in zip( state_dicts, w ) ).to( state_dicts[0][key].dtype )

        return avg


    def _val_loss( self,
                   local_val ):
        """Each center evaluates the global model on its own validation data and sends one scalar."""

        num, den = 0.0, 0
        for d in local_val.values():
            n    = d['X_train'].shape[0]
            num += n * self.global_model.loss_value( d['X_train'], d['Y_train'], self.alpha_recon, self.beta_pred )
            den += n

        return num / den


    def train( self,
               local_datasets,
               verbose   = True,
               local_val = None,
               patience  = None ):
        """
        local_val: optional validation data {center: {'X_train','Y_train'}}.  If given, the federated
        validation loss is computed after each round, the best global weights are kept (round 0 included)
        and training stops after `patience` rounds without improvement.  Details in self.history.
        """
        ids = list( local_datasets.keys() )
        n_c = [ local_datasets[c]['X_train'].shape[0] for c in ids ]

        use_es = bool( local_val ) and patience is not None
        if use_es:
            best_v, best_r, bad = self._val_loss( local_val ), 0, 0
            best_state = self.global_model.get_flat_state()

        r = -1
        for r in range( self.rounds ):

            g_state, states = self.global_model.get_flat_state(), []
            for c in ids:

                m = self.model_factory()
                m.load_flat_state( copy.deepcopy( g_state ) )
                m.train_model( X = local_datasets[c]['X_train'], Y = local_datasets[c]['Y_train'],
                               epochs = self.local_epochs, lr = self.lr, alpha_recon = self.alpha_recon,
                               beta_pred = self.beta_pred, global_state = g_state, mu_prox = self.mu_prox )
                states.append( m.get_flat_state() )

            self.global_model.load_flat_state( self._weighted_average( states, n_c ) )
            if verbose:
                print( f"  [FedAvg] round {r + 1}/{self.rounds} ({len(ids)} centers, mu_prox={self.mu_prox})" )

            if use_es:
                v = self._val_loss( local_val )
                if v < best_v - 1e-10:
                    best_v, best_r, bad = v, r + 1, 0
                    best_state = self.global_model.get_flat_state()
                else:
                    bad += 1
                    if bad >= patience:
                        break

        if use_es:
            self.global_model.load_flat_state( best_state )
            self.history = { 'best_epoch': best_r * self.local_epochs, 'epochs_run': (r + 1) * self.local_epochs,
                             'val_loss': best_v }
        else:
            self.history = { 'best_epoch': self.rounds * self.local_epochs,
                             'epochs_run': self.rounds * self.local_epochs, 'val_loss': np.nan }

        return self.global_model


def gradient_importance( nfe,
                         theta_mat,
                         X_data,
                         skip_connection = True,
                         z_mu            = None,
                         z_sd            = None ):
    """Importance_j = mean_i |d y_hat / d x_ij| * SD(x_j)   (effect of +1 SD on the prediction)."""

    theta_t = torch.FloatTensor( theta_mat.T )                   # (k, q)
    X_t     = torch.FloatTensor( X_data )
    X_t.requires_grad = True

    nfe.eval()
    Z    = nfe.encoder( X_t )
    if z_mu is not None:
        Z = ( Z - torch.FloatTensor( z_mu ) ) / torch.FloatTensor( z_sd )
    feat = torch.cat( [X_t, Z], dim = 1 ) if skip_connection else Z
    ( feat @ theta_t ).sum().backward()

    return np.mean( np.abs( X_t.grad.numpy() ), axis = 0 ) * np.std( X_data, axis = 0 )


# %% ========================================================== 5. fit + predict on a fold

def _cluster_arrays( fold ):

    return ( { c: d['X_train'] for c, d in fold['local'].items() },
             { c: d['Y_train'] for c, d in fold['local'].items() } )


def _bfi_from_cfg( feats, Ys, cfg ):

    return fit_bfi( feats, Ys,
                    lambda_local  = cfg['lambda_local'],
                    lambda_global = cfg['lambda_global'],
                    sigma2_mode   = cfg.get( 'sigma2_mode', 'global' ),
                    sigma2_bounds = tuple( cfg.get( 'sigma2_bounds', (0.02, 2.0) ) ) )


def _bfi_hetero_from_cfg( feats, Ys, cfg ):

    h = { 'lambda_s': 1.0, 'lambda_c': 1.0, 'estimate': True, 'max_iter': 200, 'tol': 1e-8 }
    h.update( cfg.get( 'hetero', {} ) )

    return fit_bfi_hetero( feats, Ys, lambda_g = cfg['lambda_global'], lambda_s = h['lambda_s'],
                           lambda_c = h['lambda_c'], estimate = h['estimate'],
                           max_iter = int( h['max_iter'] ), tol = h['tol'] )


def _bfi_fit_predict( feats,
                      Ys,
                      feat_test,
                      clusters_test,
                      cfg,
                      hetero ):
    """Fit BFI (homogeneous or heterogeneous) on precomputed features -> test predictions + extras."""

    if hetero:

        out  = _bfi_hetero_from_cfg( feats, Ys, cfg )
        pred = predict_bfi_hetero( out, feat_test, clusters_test )
        mx   = int( cfg.get( 'hetero', {} ).get( 'max_iter', 200 ) )
        return pred, { 'sigma2': out['sigma2'], 'tau2': out['tau2'], 'em_iter': out['n_iter'],
                       'em_converged': float( out['n_iter'] < mx ) }, out['beta'][1:]

    out       = _bfi_from_cfg( feats, Ys, cfg )
    theta_mat = out['theta'].reshape( Ys[next(iter(Ys))].shape[1], -1 )
    pred      = ( feat_test @ theta_mat.T )[:, 0]

    return pred, { 'sigma2': float( np.mean( list( out['sigma2'].values() ) ) ) }, theta_mat[0]


def fit_predict_bfi_linear( fold,
                            cfg,
                            hetero = False ):
    """Linear BFI: features = X.  hetero=True adds a global intercept + center intercepts."""

    feats, Ys = _cluster_arrays( fold )
    test      = fold['pooled_test']
    pred, extras, _ = _bfi_fit_predict( feats, Ys, test['X'], test['cluster'], cfg, hetero )
    # Communication rounds beyond the common standardization round (sums and sums of squares).
    # BFI-Linear with sigma2_mode='global': the RSS at the global theta can be computed on the server
    # if the message includes the scalar y'y, so it is still a single round (same numbers as in-process).
    extras['comm_rounds'] = 1.0
    extras['msg_floats']  = float( message_floats( 'BFI-Hetero' if hetero else 'BFI-Linear', fold['input_dim'] ) )

    return { 'pred': pred, 'extras': extras }


def fit_predict_bfi_fixed( fold,
                           cfg ):
    """BFI with fixed center intercepts (Jonker et al. 2024): one round, no estimate of tau^2."""

    f = { 'lambda_int': 1e-3, 'lambda_s': cfg.get( 'lambda_local', 1.0 ), 'estimate_sigma2': True }
    f.update( cfg.get( 'fixed_intercepts', {} ) )

    feats, Ys = _cluster_arrays( fold )
    msgs = { c: local_bfi_fixed_message( feats[c], Ys[c], f['lambda_s'], f['lambda_int'] ) for c in feats }
    out  = aggregate_bfi_fixed( msgs, lambda_g = cfg.get( 'lambda_global', 1.0 ), lambda_int = f['lambda_int'],
                                estimate_sigma2 = f['estimate_sigma2'] )
    test = fold['pooled_test']
    g    = np.array( list( out['gamma'].values() ) )

    return { 'pred'  : predict_bfi_fixed( out, test['X'], test['cluster'] ),
             'extras': { 'sigma2'      : out['sigma2'],
                         'gamma_var'   : float( np.var( g, ddof = 1 ) ) if len(g) > 1 else np.nan,
                         'comm_rounds' : 1.0,
                         'msg_floats'  : float( message_floats( 'BFI-Fixed', fold['input_dim'] ) ) } }


def fit_predict_dlmm( fold,
                      cfg ):
    """DLMM (Luo et al. 2022): one-shot aggregate statistics, REML on the server, prediction = fixed + BLUP."""

    reml      = cfg.get( 'dlmm', {} ).get( 'reml', True )
    feats, Ys = _cluster_arrays( fold )
    out  = fit_dlmm( { c: local_dlmm_summary( feats[c], Ys[c] ) for c in feats }, reml = reml )
    test = fold['pooled_test']
    pred = out['beta'][0] + test['X'] @ out['beta'][1:]
    pred = pred + np.array( [ out['alpha'].get( c, 0.0 ) for c in test['cluster'] ] )

    return { 'pred'  : pred,
             'extras': { 'sigma2': out['sigma2'], 'tau2': out['tau2'], 'comm_rounds': 1.0,
                         'msg_floats': float( message_floats( 'DLMM', fold['input_dim'] ) ) } }


def fit_predict_fed_em_lmm( fold,
                            cfg ):
    """
    Federated EM-LMM (iterative): distributed EM with one round per iteration and a max_rounds budget.
    Same model and priors as BFI-Hetero, so at convergence the two coincide; any difference
    measures the cost of truncation (comm_rounds = iterations performed).
    """
    h = { 'lambda_s': 1.0, 'lambda_c': 1.0 }
    h.update( cfg.get( 'hetero', {} ) )
    e = { 'max_rounds': 20, 'tol': 1e-6 }
    e.update( cfg.get( 'fed_em', {} ) )

    feats, Ys = _cluster_arrays( fold )
    msgs = { c: local_bfi_message( feats[c], Ys[c], h['lambda_s'], h['lambda_c'] ) for c in feats }
    out  = aggregate_bfi_hetero( msgs, lambda_g = cfg.get( 'lambda_global', 1.0 ), estimate = True,
                                 sigma2 = 1.0, tau2 = 1.0 / h['lambda_c'],
                                 max_iter = int( e['max_rounds'] ), tol = e['tol'] )
    test = fold['pooled_test']

    return { 'pred'  : predict_bfi_hetero( out, test['X'], test['cluster'] ),
             'extras': { 'sigma2': out['sigma2'], 'tau2': out['tau2'],
                         'comm_rounds': float( out['n_iter'] ),
                         'converged'  : float( out['n_iter'] < int( e['max_rounds'] ) ),
                         'msg_floats' : float( message_floats( 'Federated EM-LMM', fold['input_dim'] ) ) } }


def _split_local_for_validation( local,
                                 val_fraction = 0.2,
                                 seed         = 0,
                                 min_train    = 2 ):
    """For each center: round(val_fraction * n) subjects go to validation, keeping >= min_train for training."""

    rng = np.random.default_rng( seed + 104729 )
    tr, va = {}, {}
    for c, d in local.items():

        n     = d['X_train'].shape[0]
        n_val = min( int( round( val_fraction * n ) ), n - min_train )
        if n_val < 1:
            tr[c] = d
            continue
        perm   = rng.permutation( n )
        vi, ti = perm[:n_val], perm[n_val:]
        tr[c]  = { 'X_train': d['X_train'][ti], 'Y_train': d['Y_train'][ti], 'n_train': len(ti) }
        va[c]  = { 'X_train': d['X_train'][vi], 'Y_train': d['Y_train'][vi], 'n_train': len(vi) }

    return tr, va


def fit_predict_neural_bfi( fold,
                            cfg,
                            seed,
                            skip_connection = True,
                            federated       = True,
                            neural_override = None,
                            hetero          = False ):
    """
    Neural-enhanced BFI.
    skip_connection: features [X|Z] (True) or Z only (False).
    federated: extractor trained with FedAvg/FedProx on the local training data only (True, default)
               or centrally (False: a non-federated reference, not a guaranteed upper bound).
    neural_override: overrides cfg['neural'] (e.g. {'beta_pred': 0.0} -> unsupervised autoencoder).
    hetero: BFI with center intercepts (same specification as 'BFI-Hetero').

    Early stopping (cfg['early_stopping']) uses an internal validation split within each center and
    the same budget for the federated and centralized extractors: max_epochs = rounds * local_epochs,
    a check every local_epochs epochs, same patience.  Without this, the centralized extractor tends
    to overfit when centers are small and the comparison is not fair.

    Z is standardized with the training mean/SD (federated: per-center sums and sums of squares, one
    extra round of aggregate statistics) before BFI, so that the prior lambda acts on the same scale
    for X and Z.
    """
    if torch is None:
        raise ImportError( "torch is not available: cannot run Neural-BFI" )

    neural = { 'epochs': 300, 'lr': 1e-3, 'alpha_recon': 0.1, 'beta_pred': 1.0, 'hidden_dim': 16 }
    neural.update( cfg.get( 'neural', {} ) )
    neural.update( neural_override or {} )

    torch.manual_seed( seed )
    np.random.seed( seed )

    q       = fold['pooled_train']['Y'].shape[1]
    factory = lambda: NeuralFeatureExtractor( input_dim  = fold['input_dim'], latent_dim = cfg['latent_dim'],
                                              hidden_dim = neural['hidden_dim'], output_dim = q )

    fed = { 'rounds': 20, 'local_epochs': 15, 'lr': neural['lr'], 'alpha_recon': neural['alpha_recon'],
            'beta_pred': neural['beta_pred'], 'mu_prox': 0.0 }
    fed.update( cfg.get( 'federated', {} ) )

    es = { 'enabled': True, 'val_fraction': 0.2, 'patience': 3, 'min_train': 2 }
    es.update( cfg.get( 'early_stopping', {} ) )

    # Validation split taken from each center's training data (never from the fold's test set).
    # The encoder only sees the training part; the BFI head is then fitted on the whole fold training set.
    if es['enabled']:
        enc_train, enc_val = _split_local_for_validation( fold['local'], es['val_fraction'], seed, es['min_train'] )
    else:
        enc_train, enc_val = fold['local'], {}
    use_es = es['enabled'] and len( enc_val ) > 0

    if federated:

        trainer = FederatedFeatureExtractorTrainer( model_factory = factory, **fed )
        nfe     = trainer.train( enc_train, verbose = False, local_val = enc_val if use_es else None,
                                 patience = es['patience'] if use_es else None )
        hist    = trainer.history

    else:

        nfe = factory()
        if use_es:
            # same budget and checking scheme as the federated trainer
            stack = lambda dd, k: np.vstack( [ d[k] for d in dd.values() ] )
            hist  = nfe.fit_early_stopping( stack( enc_train, 'X_train' ), stack( enc_train, 'Y_train' ),
                                            stack( enc_val, 'X_train' ),   stack( enc_val, 'Y_train' ),
                                            max_epochs  = fed['rounds'] * fed['local_epochs'],
                                            check_every = fed['local_epochs'], patience = es['patience'],
                                            lr = neural['lr'], alpha_recon = neural['alpha_recon'],
                                            beta_pred = neural['beta_pred'] )
        else:
            nfe.train_model( X = fold['pooled_train']['X'], Y = fold['pooled_train']['Y'],
                             epochs = neural['epochs'], lr = neural['lr'],
                             alpha_recon = neural['alpha_recon'], beta_pred = neural['beta_pred'] )
            hist = { 'best_epoch': neural['epochs'], 'epochs_run': neural['epochs'], 'val_loss': np.nan }

    # standardize Z with aggregate training statistics (per-center sum and sum of squares)
    Z_loc = { c: nfe.extract_features( d['X_train'] ) for c, d in fold['local'].items() }
    n_tot = sum( Z.shape[0] for Z in Z_loc.values() )
    z_sum = sum( Z.sum( 0 ) for Z in Z_loc.values() )
    z_ssq = sum( (Z**2).sum( 0 ) for Z in Z_loc.values() )
    z_mu  = z_sum / n_tot
    z_sd  = np.sqrt( np.maximum( z_ssq / n_tot - z_mu**2, 0.0 ) )
    z_sd[z_sd < 1e-8] = 1.0

    def featurize( X, Z = None ):

        Z = nfe.extract_features( X ) if Z is None else Z
        Z = ( Z - z_mu ) / z_sd
        return np.hstack( [X, Z] ) if skip_connection else Z

    feats = { c: featurize( d['X_train'], Z_loc[c] ) for c, d in fold['local'].items() }
    Ys    = { c: d['Y_train']                        for c, d in fold['local'].items() }
    test  = fold['pooled_test']

    pred, extras, coef = _bfi_fit_predict( feats, Ys, featurize( test['X'] ), test['cluster'], cfg, hetero )
    extras['federated']  = float( federated )
    extras['best_epoch'] = float( hist['best_epoch'] )
    extras['epochs_run'] = float( hist['epochs_run'] )
    # FedProx rounds performed (the validation loss travels in the same round) + moments of Z + BFI message
    extras['comm_rounds'] = ( float( hist['epochs_run'] ) / fed['local_epochs'] + 2.0 ) if federated else np.nan
    if cfg.get( 'compute_importance', False ):
        extras['importance'] = gradient_importance( nfe, coef.reshape( 1, -1 ), test['X'], skip_connection,
                                                    z_mu = z_mu, z_sd = z_sd )

    return { 'pred': pred, 'extras': extras }


# %% ========================================================== 6. Baselines

class BaselineModels:
    """Baselines on a fold; predictions aligned with fold['pooled_test'] (standardized scale)."""

    def __init__( self,
                  alpha        = 1.0,
                  mlp_alpha    = 1.0,
                  mlp_hidden   = (32,),
                  mlp_max_iter = 1000,
                  tuned_alphas = tuple( np.logspace( -3, 3, 13 ) ) ):

        self.alpha, self.mlp_alpha  = alpha, mlp_alpha
        self.mlp_hidden, self.mlp_max_iter = tuple( mlp_hidden ), mlp_max_iter
        self.tuned_alphas = tuple( tuned_alphas )


    def _ridge( self, X, y, seed = 0 ):

        return Ridge( alpha = self.alpha ).fit( X, y )


    def _mlp( self, X, y, seed = 0 ):

        with warnings.catch_warnings():
            warnings.simplefilter( 'ignore', ConvergenceWarning )
            return MLPRegressor( hidden_layer_sizes = self.mlp_hidden, max_iter = self.mlp_max_iter,
                                 alpha = self.mlp_alpha, random_state = seed ).fit( X, y )


    @staticmethod
    def _local( fold, fit_fn, seed ):
        """One model per center (no federation); predictions assembled in test order."""

        test = fold['pooled_test']
        pred = np.full( test['X'].shape[0], np.nan )
        for c, d in fold['local'].items():

            mask = test['cluster'] == c
            if mask.any():
                pred[mask] = fit_fn( d['X_train'], d['Y_train'].ravel(), seed ).predict( test['X'][mask] )

        return { 'pred': pred, 'extras': { 'comm_rounds': 0.0 } }     # no communication


    @staticmethod
    def _pooled( fold, fit_fn, seed ):

        tr = fold['pooled_train']

        return { 'pred': fit_fn( tr['X'], tr['Y'].ravel(), seed ).predict( fold['pooled_test']['X'] ), 'extras': {} }


    def _ridge_tuned( self, X, y, seed = 0 ):
        """Local ridge with the penalty chosen internally (efficient GCV/leave-one-out) on the center's training data only."""

        return RidgeCV( alphas = self.tuned_alphas ).fit( X, y )


    def local_ridge ( self, fold, seed = 0 ): return self._local ( fold, self._ridge, seed )
    def local_ridge_tuned( self, fold, seed = 0 ): return self._local( fold, self._ridge_tuned, seed )
    def pooled_ridge( self, fold, seed = 0 ): return self._pooled( fold, self._ridge, seed )
    def local_mlp   ( self, fold, seed = 0 ): return self._local ( fold, self._mlp,   seed )
    def pooled_mlp  ( self, fold, seed = 0 ): return self._pooled( fold, self._mlp,   seed )


    def mixed_effects( self,
                       fold ):
        """
        Random-intercept LMM by cluster (REML): y = X beta + u_c + eps.
        Prediction = fixed effects + center BLUP.  If the fit fails it falls back to pooled ridge
        (extras['fallback'] = 1).  Not a federated model: used as an "oracle" reference.
        extras['sm_converged'] stores the convergence flag reported by statsmodels; with very small
        centers it can be False (DLMM, which optimizes the same REML criterion, then reaches a
        restricted log-likelihood >= the statsmodels one).
        """
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM

        tr, te = fold['pooled_train'], fold['pooled_test']
        Xtr    = sm.add_constant( tr['X'], has_constant = 'add' )
        Xte    = sm.add_constant( te['X'], has_constant = 'add' )

        try:

            with warnings.catch_warnings():
                warnings.simplefilter( 'ignore' )
                # default optimizer chain (bfgs -> lbfgs -> cg): forcing 'lbfgs' converges to the boundary (var = 0)
                res = MixedLM( endog = tr['Y'].ravel(), exog = Xtr, groups = tr['cluster'] ).fit( reml = True )

            pred, singular = Xte @ np.asarray( res.fe_params ), 0.0
            try:
                for c, re in res.random_effects.items():

                    mask = te['cluster'] == c
                    if mask.any():
                        pred[mask] += float( np.asarray(re)[0] )
            except ValueError:
                singular = 1.0                       # intercept variance ~0: fixed effects only

            if not np.all( np.isfinite(pred) ):
                raise RuntimeError( "non-finite predictions" )

            return { 'pred': pred, 'extras': { 'fallback': 0.0, 'singular_re': singular,
                                                'sm_converged': float( bool( getattr( res, 'converged', True ) ) ),
                                                'var_re': float( np.asarray(res.cov_re)[0, 0] ),
                                                'sigma2': float( res.scale ) } }

        except Exception:

            out = self.pooled_ridge( fold )
            out['extras'] = { 'fallback': 1.0 }

            return out


# %%
