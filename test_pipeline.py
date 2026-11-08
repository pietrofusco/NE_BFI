#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline tests.  Run with `python test_pipeline.py` (or pytest) from the project root.
Neural tests need torch and LMM tests need statsmodels; they are skipped if those are missing.
"""
import importlib.util
import numpy as np
from scipy.optimize       import minimize
from sklearn.linear_model import Ridge
from models               import ( LocalBayesianGLM, fit_bfi, fit_predict_bfi_linear, fit_bfi_hetero,
                                  fit_predict_neural_bfi,
                                  local_bfi_message, aggregate_bfi_hetero, _sufficient_from_message )
from data_handler         import make_repeated_cv_splits, build_fold
from simulation           import simulate_dataset
from evaluation           import ( CVResults, bootstrap_all, summarize, paired_comparisons, CVRunner,
                                  build_registry, point_metrics )

HAS_STATSMODELS = importlib.util.find_spec( 'statsmodels' ) is not None

def _fold( seed = 0, nonlin = 0.0, n_per_cluster = 40 ):

    raw    = simulate_dataset( n_clusters = 8, n_per_cluster = n_per_cluster, nonlin = nonlin, seed = seed )
    splits = make_repeated_cv_splits( raw.clusters, 5, 1, seed = seed )
    s      = splits[0]

    return raw, build_fold( raw, s['train_idx'], s['test_idx'], seed = seed )


def test_analytic_map_and_hessian_match_numeric():

    rng = np.random.default_rng( 0 )
    Z, Y = rng.normal( size = (30, 5) ), rng.normal( size = (30, 2) )
    m = LocalBayesianGLM( Y, Z, lambda_reg = 1.3, sigma2 = 0.7 )
    theta, A = m.fit()

    res = minimize( m._negative_log_posterior, np.zeros( m.dim_theta ), method = 'L-BFGS-B',
                    options = { 'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-11 } )
    assert np.max( np.abs( res.x - theta ) ) < 1e-5

    eps, H = 1e-4, np.zeros( (m.dim_theta, m.dim_theta) )
    f = m._negative_log_posterior
    for i in range( m.dim_theta ):
        for j in range( m.dim_theta ):
            e_i, e_j = np.eye( m.dim_theta )[i] * eps, np.eye( m.dim_theta )[j] * eps
            H[i, j]  = ( f(theta+e_i+e_j) - f(theta+e_i-e_j) - f(theta-e_i+e_j) + f(theta-e_i-e_j) ) / (4*eps**2)
    assert np.max( np.abs( H - A ) ) < 1e-4


def test_bfi_linear_equals_pooled_ridge():

    _, fold = _fold()
    alpha = 1.7
    cfg   = { 'lambda_local': alpha, 'lambda_global': alpha, 'sigma2_mode': 'fixed' }

    pred_bfi   = fit_predict_bfi_linear( fold, cfg )['pred']
    tr         = fold['pooled_train']
    pred_ridge = Ridge( alpha = alpha ).fit( tr['X'], tr['Y'].ravel() ).predict( fold['pooled_test']['X'] )
    assert np.max( np.abs( pred_bfi - pred_ridge ) ) < 1e-8, "linear BFI (sigma2=1) != pooled ridge"

    # with estimated sigma^2: equivalent to ridge with alpha = sigma^2 * lambda
    cfg2 = { 'lambda_local': 1.0, 'lambda_global': 1.0, 'sigma2_mode': 'global' }
    out  = fit_predict_bfi_linear( fold, cfg2 )
    s2   = out['extras']['sigma2']
    pred_ridge2 = Ridge( alpha = s2 * 1.0 ).fit( tr['X'], tr['Y'].ravel() ).predict( fold['pooled_test']['X'] )
    assert np.max( np.abs( out['pred'] - pred_ridge2 ) ) < 1e-8
    assert 0.02 <= s2 <= 2.0


def test_old_aggregation_over_regularizes():
    """A naive aggregation that counts the global prior L times does not reproduce pooled ridge."""

    _, fold = _fold()
    feats = { c: d['X_train'] for c, d in fold['local'].items() }
    Ys    = { c: d['Y_train'] for c, d in fold['local'].items() }
    out   = fit_bfi( feats, Ys, 1.0, 1.0, sigma2_mode = 'fixed' )

    theta_old_num, A_old = 0, 0
    for r in out['local_results']:
        A_old         = A_old + r['A_hat'] + ( 1.0 - r['lambda_reg'] ) * np.eye( len( r['theta_hat'] ) )
        theta_old_num = theta_old_num + r['A_hat'] @ r['theta_hat']
    theta_old = np.linalg.solve( A_old, theta_old_num )

    tr = fold['pooled_train']
    ridge = Ridge( alpha = 1.0 ).fit( tr['X'], tr['Y'].ravel() )
    assert np.max( np.abs( out['theta'] - ridge.coef_ ) ) < 1e-8
    assert np.max( np.abs( theta_old - ridge.coef_ ) ) > 1e-3


def test_preprocessing_has_no_leakage():

    raw, fold = _fold( seed = 3 )
    tr_idx    = fold['pooled_train']['idx']
    te_idx    = fold['pooled_test']['idx']

    # (a) statistics depend on the training data only
    pp = fold['preproc']
    assert np.allclose( pp.scaler_Y.mean_, raw.Y[tr_idx].mean( axis = 0 ) )
    assert np.allclose( pp.mean_X[:7], raw.X[tr_idx][:, :7].mean( axis = 0 ) )

    # (b) perturbing the test values changes nothing in preprocessing/training
    raw2      = simulate_dataset( n_clusters = 8, n_per_cluster = 40, nonlin = 0.0, seed = 3 )
    raw2.X[te_idx] += 1000.0
    raw2.Y[te_idx] += 1000.0
    fold2 = build_fold( raw2, tr_idx, te_idx, seed = 3 )
    assert np.allclose( fold['pooled_train']['X'], fold2['pooled_train']['X'] )
    assert np.allclose( fold['pooled_train']['Y'], fold2['pooled_train']['Y'] )
    assert np.allclose( fold['preproc'].mean_X, fold2['preproc'].mean_X )


def test_cv_splits_cover_each_subject_once_per_repeat():

    raw    = simulate_dataset( n_clusters = 6, n_per_cluster = 25, seed = 1 )
    splits = make_repeated_cv_splits( raw.clusters, 5, 4, seed = 0 )
    assert len( splits ) == 20
    for r in range( 4 ):
        seen = np.concatenate( [ s['test_idx'] for s in splits if s['repeat'] == r ] )
        assert len( seen ) == raw.n and len( np.unique( seen ) ) == raw.n
    for s in splits:
        assert len( np.intersect1d( s['train_idx'], s['test_idx'] ) ) == 0
        assert set( np.unique( raw.clusters[s['test_idx']] ) ) <= set( np.unique( raw.clusters[s['train_idx']] ) )


def test_bootstrap_paired_comparisons():

    rng = np.random.default_rng( 0 )
    N, R = 300, 4
    cl   = np.repeat( np.arange( 10 ), 30 )
    y    = rng.normal( size = N )
    good = y + rng.normal( 0, 0.5, size = (R, N) )
    bad  = y + rng.normal( 0, 1.0, size = (R, N) )
    res  = CVResults( y_true = y, clusters = cl,
                      preds = { 'ref': bad, 'same': bad.copy(), 'good': good },
                      fold_mse = { 'ref': rng.uniform( 1, 1.1, 20 ), 'same': rng.uniform( 1, 1.1, 20 ),
                                   'good': rng.uniform( 0.25, 0.3, 20 ) },
                      n_train = np.full( 20, 240 ), n_test = np.full( 20, 60 ) )
    boot = bootstrap_all( res, n_boot = 300, unit = 'cluster', seed = 1 )
    comp = paired_comparisons( res, boot, 'ref' ).set_index( 'Model' )
    assert abs( comp.loc['same', 'dR2'] ) < 1e-12 and comp.loc['same', 'dR2_lo'] == comp.loc['same', 'dR2_hi'] == 0
    assert comp.loc['good', 'dR2_lo'] > 0 and comp.loc['good', 'p_bootstrap'] < 0.05
    assert comp.loc['good', 'p_NB'] < 0.05
    summ = summarize( res, boot ).set_index( 'Model' )
    assert summ.loc['good', 'R2_lo'] <= summ.loc['good', 'R2'] <= summ.loc['good', 'R2_hi']


def test_end_to_end_without_torch():

    raw = simulate_dataset( n_clusters = 10, n_per_cluster = 35, nonlin = 0.0, snr = 1.5, seed = 5 )
    cfg = { 'lambda_local': 1.0, 'lambda_global': 1.0, 'sigma2_mode': 'global',
            'cv': { 'n_splits': 5, 'n_repeats': 2, 'seed': 0 }, 'mlp_max_iter': 300 }
    names = [ 'Ridge (local)', 'Ridge (pooled)', 'MLP (local)', 'MLP (pooled)', 'BFI-Linear', 'BFI-Hetero' ]
    if HAS_STATSMODELS:
        names.append( 'Mixed-effects LMM' )
    reg = build_registry( cfg, only = names )
    res = CVRunner( raw, cfg, reg ).run( verbose = False )
    boot = bootstrap_all( res, n_boot = 100, seed = 0 )
    summ = summarize( res, boot ).set_index( 'Model' )
    assert summ.loc['Ridge (pooled)', 'R2'] > 0.2
    # linear BFI with estimated sigma2 ~ pooled ridge
    assert abs( summ.loc['BFI-Linear', 'R2'] - summ.loc['Ridge (pooled)', 'R2'] ) < 0.02
    if HAS_STATSMODELS:
        fb = sum( e['fallback'] for e in res.extras['Mixed-effects LMM'] )
        assert fb == 0, f"LMM used the fallback in {fb} folds"


def test_lmm_exploits_cluster_effect():
    """Regression test: with a strong center effect the LMM must beat pooled ridge (no zero variance)."""

    if not HAS_STATSMODELS:
        print( "  (skipped: statsmodels not available)" )
        return

    raw = simulate_dataset( n_clusters = 15, n_per_cluster = 40, nonlin = 0.0, tau = 1.2,
                            slope_het = 0.0, covariate_shift = 0.0, snr = 2.0, seed = 7 )
    cfg = { 'lambda_local': 1.0, 'lambda_global': 1.0, 'cv': { 'n_splits': 5, 'n_repeats': 2, 'seed': 0 } }
    reg = build_registry( cfg, only = [ 'Ridge (pooled)', 'Mixed-effects LMM' ] )
    res = CVRunner( raw, cfg, reg ).run( verbose = False )
    boot = bootstrap_all( res, n_boot = 50, seed = 0 )
    summ = summarize( res, boot ).set_index( 'Model' )
    assert summ.loc['Mixed-effects LMM', 'R2'] > summ.loc['Ridge (pooled)', 'R2'] + 0.05
    v = [ e['var_re'] for e in res.extras['Mixed-effects LMM'] ]
    assert min( v ) > 0.05, f"degenerate random-intercept variance: {v}"


def test_neural_bfi_smoke():

    try:
        import torch  # noqa
    except Exception:
        print( "  (skipped: torch not available)" )
        return

    raw = simulate_dataset( n_clusters = 8, n_per_cluster = 30, nonlin = 0.5, seed = 2 )
    cfg = { 'latent_dim': 4, 'lambda_local': 1.0, 'lambda_global': 1.0, 'sigma2_mode': 'global',
            'neural': { 'epochs': 60 }, 'federated': { 'rounds': 2, 'local_epochs': 3, 'mu_prox': 0.01 },
            'compute_importance': True, 'cv': { 'n_splits': 5, 'n_repeats': 1, 'seed': 0 },
            'neural_variants': { 'NF'  : { 'skip': True, 'federated': True,  'hetero': True },
                                 'NFz' : { 'skip': False, 'federated': True, 'hetero': False },
                                 'NC'  : { 'skip': True, 'federated': False, 'hetero': True } } }
    reg = build_registry( cfg, only = [ 'BFI-Hetero', 'NF', 'NFz', 'NC' ] )
    res = CVRunner( raw, cfg, reg ).run( verbose = False )
    for m, P in res.preds.items():
        assert np.isfinite( P ).all(), m
    assert res.extras['NF'][0]['federated'] == 1.0 and res.extras['NC'][0]['federated'] == 0.0
    assert 'importance' in res.extras['NF'][0] and 'tau2' in res.extras['NF'][0]


def test_default_neural_variants_are_federated():

    from evaluation import DEFAULT_NEURAL_VARIANTS
    assert all( v.get( 'federated', True ) for v in DEFAULT_NEURAL_VARIANTS.values() )


# ------------------------------------------------------------------ heterogeneous BFI

def _penalized_ridge_with_center_dummies( fold, lambda_b0, lambda_g, lambda_c ):
    """Pooled (non-federated) closed-form reference solution."""

    tr  = fold['pooled_train']
    ids = sorted( fold['local'].keys() )
    D   = ( tr['cluster'][:, None] == np.array( ids )[None, :] ).astype( float )
    W   = np.hstack( [ np.ones( (len(D), 1) ), tr['X'], D ] )
    P   = np.diag( np.r_[ lambda_b0, np.full( tr['X'].shape[1], lambda_g ), np.full( len(ids), lambda_c ) ] )
    th  = np.linalg.solve( W.T @ W + P, W.T @ tr['Y'].ravel() )

    return th, ids


def test_bfi_hetero_fixed_equals_pooled_penalized_ridge():
    """Without estimation (sigma^2 = 1, tau^2 = 1/lambda_c) the federated heterogeneous BFI is exact."""

    _, fold = _fold( seed = 4, n_per_cluster = 15 )
    feats   = { c: d['X_train'] for c, d in fold['local'].items() }
    Ys      = { c: d['Y_train'] for c, d in fold['local'].items() }
    for lam_s in ( 0.3, 5.0 ):                          # the local prior must have no effect
        out = fit_bfi_hetero( feats, Ys, lambda_g = 1.7, lambda_s = lam_s, lambda_c = 2.5, estimate = False )
        th, ids = _penalized_ridge_with_center_dummies( fold, 1e-6, 1.7, 2.5 )
        p = 1 + fold['input_dim']
        assert np.max( np.abs( out['beta'] - th[:p] ) ) < 1e-8
        assert np.max( np.abs( np.array( [ out['alpha'][c] for c in ids ] ) - th[p:] ) ) < 1e-8


def test_bfi_hetero_message_recovers_sufficient_statistics():

    rng = np.random.default_rng( 1 )
    X, y = rng.normal( size = (7, 4) ), rng.normal( size = 7 )          # center with n = 7
    msg  = local_bfi_message( X, y, lambda_s = 0.7, lambda_c = 3.0 )
    S, s, yy = _sufficient_from_message( msg )
    W = np.hstack( [ np.ones( (7, 1) ), X, np.ones( (7, 1) ) ] )
    assert np.allclose( S, W.T @ W ) and np.allclose( s, W.T @ y ) and np.isclose( yy, y @ y )


def test_bfi_hetero_em_equals_lmm_reml():
    """As lambda_g -> 0 the EM estimate (flat prior on beta, which is integrated out) matches pooled
    MixedLM REML: fixed effects, sigma^2, tau^2 and center BLUPs."""

    if not HAS_STATSMODELS:
        print( "  (skipped: statsmodels not available)" )
        return
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM

    raw = simulate_dataset( n_clusters = 12, n_per_cluster = 12, nonlin = 0.0, tau = 1.0,
                            slope_het = 0.0, snr = 2.0, seed = 11 )
    s    = make_repeated_cv_splits( raw.clusters, 5, 1, seed = 0 )[0]
    fold = build_fold( raw, s['train_idx'], s['test_idx'] )
    feats = { c: d['X_train'] for c, d in fold['local'].items() }
    Ys    = { c: d['Y_train'] for c, d in fold['local'].items() }
    out   = aggregate_bfi_hetero( { c: local_bfi_message( feats[c], Ys[c] ) for c in feats },
                                  lambda_g = 1e-9, lambda_b0 = 1e-9, max_iter = 20000, tol = 1e-13 )

    tr  = fold['pooled_train']
    lmm = MixedLM( tr['Y'].ravel(), sm.add_constant( tr['X'] ), groups = tr['cluster'] ).fit( reml = True )
    assert np.max( np.abs( out['beta'] - np.asarray( lmm.fe_params ) ) ) < 1e-5
    assert abs( out['sigma2'] - lmm.scale ) < 1e-5
    assert abs( out['tau2'] - float( np.asarray( lmm.cov_re )[0, 0] ) ) < 1e-5
    re = { c: float( np.asarray( v )[0] ) for c, v in lmm.random_effects.items() }
    assert max( abs( out['alpha'][c] - re[c] ) for c in re ) < 1e-5


def test_bfi_hetero_beats_homogeneous_with_center_effect():

    raw = simulate_dataset( n_clusters = 20, n_per_cluster = 10, nonlin = 0.0, tau = 1.2,
                            slope_het = 0.0, covariate_shift = 0.0, snr = 2.0, seed = 7 )
    cfg = { 'lambda_local': 1.0, 'lambda_global': 1.0, 'cv': { 'n_splits': 5, 'n_repeats': 2, 'seed': 0 } }
    reg = build_registry( cfg, only = [ 'BFI-Linear', 'BFI-Hetero' ] )
    res = CVRunner( raw, cfg, reg ).run( verbose = False )
    r2  = { m: point_metrics( res.y_true, P )['R2'] for m, P in res.preds.items() }
    assert r2['BFI-Hetero'] > r2['BFI-Linear'] + 0.05, r2


# ------------------------------------------------------------------ early stopping, variance components, simulation

def test_early_stopping_equal_budget():

    try:
        import torch  # noqa
    except Exception:
        print( "  (skipped: torch not available)" )
        return

    _, fold = _fold( seed = 6, nonlin = 0.3, n_per_cluster = 20 )
    cfg = { 'latent_dim': 4, 'lambda_local': 1.0, 'lambda_global': 1.0, 'neural': { 'epochs': 999 },
            'federated': { 'rounds': 4, 'local_epochs': 5, 'mu_prox': 0.01 },
            'early_stopping': { 'enabled': True, 'val_fraction': 0.2, 'patience': 2 } }
    budget = 4 * 5
    for fed in ( True, False ):
        ex = fit_predict_neural_bfi( fold, cfg, 0, federated = fed, hetero = True )['extras']
        assert 0 <= ex['best_epoch'] <= ex['epochs_run'] <= budget, ( fed, ex )
        assert ex['best_epoch'] % 5 == 0                 # same checking scheme
    cfg['early_stopping'] = { 'enabled': False }
    ex = fit_predict_neural_bfi( fold, cfg, 0, federated = False, hetero = True )['extras']
    assert ex['epochs_run'] == 999                       # without early stopping: full epochs


def test_validation_split_is_inside_training():

    from models import _split_local_for_validation
    _, fold = _fold( seed = 2, n_per_cluster = 6 )
    tr, va = _split_local_for_validation( fold['local'], 0.2, 0, 2 )
    for c, d in fold['local'].items():
        n_tr = tr[c]['X_train'].shape[0]
        n_va = va[c]['X_train'].shape[0] if c in va else 0
        assert n_tr + n_va == d['n_train'] and n_tr >= 2
        if c in va:                                       # disjoint rows, all taken from training
            rows = { tuple( r ) for r in d['X_train'] }
            assert all( tuple( r ) in rows for r in va[c]['X_train'] )


def test_variance_components_agree_hetero_lmm():

    if not HAS_STATSMODELS:
        print( "  (skipped: statsmodels not available)" )
        return
    from evaluation import variance_components
    raw = simulate_dataset( n_clusters = 15, n_per_cluster = 20, nonlin = 0.0, tau = 0.8, seed = 9 )
    cfg = { 'lambda_local': 1.0, 'lambda_global': 1.0, 'cv': { 'n_splits': 5, 'n_repeats': 1, 'seed': 0 } }
    res = CVRunner( raw, cfg, build_registry( cfg, only = [ 'BFI-Hetero', 'Mixed-effects LMM' ] ) ).run( verbose = False )
    vc  = variance_components( res ).set_index( 'Model' )
    assert set( vc.index ) == { 'BFI-Hetero', 'Mixed-effects LMM' }
    assert 0 < vc.loc['BFI-Hetero', 'ICC'] < 1
    assert abs( vc.loc['BFI-Hetero', 'ICC'] - vc.loc['Mixed-effects LMM', 'ICC'] ) < 0.02


def test_simulation_mcse_and_resume():

    import tempfile, pathlib
    from simulation import run_simulation
    cfg = { 'lambda_local': 1.0, 'lambda_global': 1.0,
            'simulation': { 'tau': [0.8], 'nonlin': [0.0], 'n_per_cluster': [5, 20], 'n_clusters': 8,
                            'n_datasets': 3, 'n_datasets_neural': 0, 'n_jobs': 1,
                            'models_linear': [ 'Ridge (local)', 'BFI-Linear', 'BFI-Hetero' ],
                            'models_neural': [], 'reference': 'BFI-Hetero' } }
    with tempfile.TemporaryDirectory() as d:
        raw, summ, pr = run_simulation( cfg, out_dir = d, verbose = False )
        assert len( list( ( pathlib.Path( d ) / 'parts' ).glob( '*.csv' ) ) ) == 2
        assert ( summ['n_rep'] == 3 ).all() and summ['R2_mcse'].notna().all()
        assert set( pr['model'] ) == { 'Ridge (local)', 'BFI-Linear' } and pr['dR2_mcse'].notna().all()
        assert ( summ['n_per_cluster'].isin( [5, 20] ) ).all()
        assert raw.groupby( 'n_per_cluster' )['n_total'].min().loc[5] >= 8 * 5      # at least 5 per center
        raw2, _, _ = run_simulation( cfg, out_dir = d, verbose = False )             # resume: reads saved parts
        assert np.allclose( raw.sort_values( ['n_per_cluster', 'rep', 'model'] )['R2'].values,
                            raw2.sort_values( ['n_per_cluster', 'rep', 'model'] )['R2'].values )
        # no neural model -> paired table has linear models only; figures are still produced
        import reporting
        reporting.plot_simulation( summ, pr, d )
        assert ( pathlib.Path( d ) / 'simulation_paired.png' ).exists()


def test_report_figures_smoke():

    import tempfile, pathlib, reporting
    raw = simulate_dataset( n_clusters = 6, n_per_cluster = 15, seed = 1 )
    cfg = { 'lambda_local': 1.0, 'lambda_global': 1.0, 'cv': { 'n_splits': 5, 'n_repeats': 1, 'seed': 0 },
            'learning_curve': { 'fractions': [0.5, 1.0], 'n_repeats': 1, 'n_boot': 20 } }
    reg = build_registry( cfg, only = [ 'Ridge (local)', 'BFI-Linear', 'BFI-Hetero' ] )
    res = CVRunner( raw, cfg, reg ).run( verbose = False )
    boot = bootstrap_all( res, n_boot = 20 )
    from evaluation import run_learning_curve
    lcs, lcd = run_learning_curve( raw, cfg, reg, verbose = False )
    with tempfile.TemporaryDirectory() as d:
        reporting.plot_forest( summarize( res, boot ), d, 'T' )
        reporting.plot_paired( paired_comparisons( res, boot, 'BFI-Hetero' ), d, 'T' )
        reporting.plot_learning_curves( lcs, lcd, d, 'T' )
        assert all( ( pathlib.Path( d ) / f ).exists() for f in
                    [ 'forest_metrics.png', 'paired_differences.png', 'learning_curves.png' ] )


# ------------------------------------------------------------------ comparators

def _pooled_dummy_regression_no_intercept( fold, lambda_g, lambda_int ):
    """Pooled reference for BFI-Fixed: y = X beta + D gamma, ridge prior, sigma^2 = 1."""

    tr  = fold['pooled_train']
    ids = list( fold['local'].keys() )
    D   = ( tr['cluster'][:, None] == np.array( ids )[None, :] ).astype( float )
    W   = np.hstack( [ tr['X'], D ] )
    P   = np.diag( np.r_[ np.full( tr['X'].shape[1], lambda_g ), np.full( len(ids), lambda_int ) ] )

    return np.linalg.solve( W.T @ W + P, W.T @ tr['Y'].ravel() ), ids


def test_bfi_fixed_equals_pooled_dummy_regression():
    """BFI formula with center-specific parameters (Jonker 2024) = pooled regression with center dummies."""

    from models import local_bfi_fixed_message, aggregate_bfi_fixed
    _, fold = _fold( seed = 4, n_per_cluster = 12 )
    feats   = { c: d['X_train'] for c, d in fold['local'].items() }
    Ys      = { c: d['Y_train'] for c, d in fold['local'].items() }
    for lam_s in ( 0.2, 4.0 ):                                    # the local prior must have no effect
        msgs = { c: local_bfi_fixed_message( feats[c], Ys[c], lambda_s = lam_s, lambda_int = 0.1 ) for c in feats }
        out  = aggregate_bfi_fixed( msgs, lambda_g = 1.3, lambda_int = 0.1, estimate_sigma2 = False )
        th, ids = _pooled_dummy_regression_no_intercept( fold, 1.3, 0.1 )
        p = fold['input_dim']
        assert np.max( np.abs( out['beta'] - th[:p] ) ) < 1e-8
        assert np.max( np.abs( np.array( [ out['gamma'][c] for c in ids ] ) - th[p:] ) ) < 1e-8


def test_bfi_fixed_sigma2_matches_pooled_fixed_point():
    """sigma^2 estimated on the server from the messages = fixed point RSS / (N - p_eff) on the pooled data."""

    from models import local_bfi_fixed_message, aggregate_bfi_fixed
    _, fold = _fold( seed = 8, n_per_cluster = 20 )
    feats = { c: d['X_train'] for c, d in fold['local'].items() }
    Ys    = { c: d['Y_train'] for c, d in fold['local'].items() }
    out   = aggregate_bfi_fixed( { c: local_bfi_fixed_message( feats[c], Ys[c] ) for c in feats },
                                 lambda_g = 1.0, lambda_int = 1e-3, tol = 1e-14, max_iter = 500 )
    tr  = fold['pooled_train']
    ids = list( fold['local'].keys() )
    W   = np.hstack( [ tr['X'], ( tr['cluster'][:, None] == np.array( ids )[None, :] ).astype( float ) ] )
    Lam = np.r_[ np.full( tr['X'].shape[1], 1.0 ), np.full( len(ids), 1e-3 ) ]
    s2  = out['sigma2']
    th  = np.linalg.solve( W.T @ W + s2 * np.diag( Lam ), W.T @ tr['Y'].ravel() )
    H   = W @ np.linalg.solve( W.T @ W + s2 * np.diag( Lam ), W.T )
    rss = np.sum( ( tr['Y'].ravel() - W @ th )**2 )
    assert abs( s2 - rss / ( W.shape[0] - np.trace( H ) ) ) < 1e-8


def test_bfi_fixed_matches_hetero_without_shrinkage():
    """With a flat prior on the intercepts (lambda_int -> 0, tau^2 -> inf) BFI-Fixed and BFI-Hetero give the same predictions."""

    from models import ( local_bfi_fixed_message, aggregate_bfi_fixed, predict_bfi_fixed,
                         predict_bfi_hetero )
    _, fold = _fold( seed = 5, n_per_cluster = 15 )
    feats = { c: d['X_train'] for c, d in fold['local'].items() }
    Ys    = { c: d['Y_train'] for c, d in fold['local'].items() }
    te    = fold['pooled_test']
    fx = aggregate_bfi_fixed( { c: local_bfi_fixed_message( feats[c], Ys[c], 1.0, 1e-9 ) for c in feats },
                              lambda_g = 1.0, lambda_int = 1e-9, estimate_sigma2 = False )
    ht = fit_bfi_hetero( feats, Ys, lambda_g = 1.0, lambda_c = 1e-9, estimate = False )
    p1 = predict_bfi_fixed( fx, te['X'], te['cluster'] )
    p2 = predict_bfi_hetero( ht, te['X'], te['cluster'] )
    assert np.max( np.abs( p1 - p2 ) ) < 1e-5


def test_dlmm_equals_lmm_reml_and_ml():
    """DLMM (one-shot aggregate statistics) = pooled MixedLM: fixed effects, sigma^2, tau^2, BLUPs."""

    if not HAS_STATSMODELS:
        print( "  (skipped: statsmodels not available)" )
        return
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    from models import local_dlmm_summary, fit_dlmm

    raw = simulate_dataset( n_clusters = 12, n_per_cluster = 12, nonlin = 0.0, tau = 1.0,
                            slope_het = 0.0, snr = 2.0, seed = 11 )
    s    = make_repeated_cv_splits( raw.clusters, 5, 1, seed = 0 )[0]
    fold = build_fold( raw, s['train_idx'], s['test_idx'] )
    summ = { c: local_dlmm_summary( d['X_train'], d['Y_train'] ) for c, d in fold['local'].items() }
    tr   = fold['pooled_train']
    for reml in ( True, False ):
        out = fit_dlmm( summ, reml = reml )
        # for ML the default statsmodels optimizer stops early (log-likelihood ~1e-8 below DLMM's);
        # Powell reaches the same optimum
        lmm = MixedLM( tr['Y'].ravel(), sm.add_constant( tr['X'] ), groups = tr['cluster'] ).fit(
                  reml = reml, **( {} if reml else { 'method': 'powell', 'maxiter': 5000 } ) )
        assert np.max( np.abs( out['beta'] - np.asarray( lmm.fe_params ) ) ) < 1e-5, reml
        assert abs( out['sigma2'] - lmm.scale ) < 1e-5, reml
        assert abs( out['tau2'] - float( np.asarray( lmm.cov_re )[0, 0] ) ) < 1e-5, reml
        re = { c: float( np.asarray( v )[0] ) for c, v in lmm.random_effects.items() }
        assert max( abs( out['alpha'][c] - re[c] ) for c in re ) < 1e-5, reml


def test_dlmm_equals_bfi_hetero_with_flat_prior():
    """Two independent one-shot algorithms (EM on BFI messages; direct maximization on aggregate
    statistics) reach the same REML solution."""

    from models import local_dlmm_summary, fit_dlmm
    _, fold = _fold( seed = 12, n_per_cluster = 10 )
    feats = { c: d['X_train'] for c, d in fold['local'].items() }
    Ys    = { c: d['Y_train'] for c, d in fold['local'].items() }
    dl = fit_dlmm( { c: local_dlmm_summary( feats[c], Ys[c] ) for c in feats } )
    ht = aggregate_bfi_hetero( { c: local_bfi_message( feats[c], Ys[c] ) for c in feats },
                               lambda_g = 1e-9, lambda_b0 = 1e-9, max_iter = 20000, tol = 1e-13 )
    assert np.max( np.abs( dl['beta'] - ht['beta'] ) ) < 1e-5
    assert abs( dl['tau2'] - ht['tau2'] ) < 1e-5 and abs( dl['sigma2'] - ht['sigma2'] ) < 1e-5


def test_communication_rounds_and_message_size():
    """BFI-Linear, BFI-Fixed, BFI-Hetero and DLMM: 1 round; iterative EM: > 1 round; local: 0; pooled: NaN."""

    from evaluation import communication_summary
    raw = simulate_dataset( n_clusters = 10, n_per_cluster = 15, tau = 0.3, seed = 3 )
    cfg = { 'lambda_local': 1.0, 'lambda_global': 1.0, 'cv': { 'n_splits': 5, 'n_repeats': 1, 'seed': 0 },
            'fed_em': { 'max_rounds': 20, 'tol': 1e-6 } }
    names = [ 'Ridge (local)', 'Ridge (pooled)', 'BFI-Linear', 'BFI-Fixed', 'BFI-Hetero', 'DLMM',
              'Federated EM-LMM (iterative)' ]
    res  = CVRunner( raw, cfg, build_registry( cfg, only = names ) ).run( verbose = False )
    comm = communication_summary( res ).set_index( 'Model' )
    for m in ( 'BFI-Linear', 'BFI-Fixed', 'BFI-Hetero', 'DLMM' ):
        assert comm.loc[m, 'rounds_min'] == comm.loc[m, 'rounds_max'] == 1.0, m
    assert comm.loc['Federated EM-LMM (iterative)', 'rounds_min'] > 1.0
    assert comm.loc['Federated EM-LMM (iterative)', 'rounds_max'] <= 20.0
    assert comm.loc['Ridge (local)', 'rounds_mean'] == 0.0 and not comm.loc['Ridge (local)', 'federated']
    assert np.isnan( comm.loc['Ridge (pooled)', 'rounds_mean'] )
    p = raw.input_dim
    assert comm.loc['BFI-Hetero', 'msg_floats'] == ( p + 2 ) + ( p + 2 ) * ( p + 3 ) // 2 + 2


def test_fed_em_with_large_budget_equals_bfi_hetero():
    """Iterative EM, if allowed to converge, matches BFI-Hetero: only the number of rounds differs."""

    raw = simulate_dataset( n_clusters = 10, n_per_cluster = 12, tau = 0.5, seed = 4 )
    cfg = { 'lambda_local': 1.0, 'lambda_global': 1.0, 'cv': { 'n_splits': 5, 'n_repeats': 1, 'seed': 0 },
            'fed_em': { 'max_rounds': 200, 'tol': 1e-8 } }
    res = CVRunner( raw, cfg, build_registry( cfg, only = [ 'BFI-Hetero', 'Federated EM-LMM (iterative)' ] ) ).run( verbose = False )
    assert np.max( np.abs( res.preds['BFI-Hetero'] - res.preds['Federated EM-LMM (iterative)'] ) ) < 1e-6
    assert all( e['converged'] == 1.0 for e in res.extras['Federated EM-LMM (iterative)'] )


def test_fixed_intercepts_overfit_with_tiny_centers():
    """Regression test: with 5-patient centers and low ICC, fixed intercepts (estimated from ~4 patients) lose
    to random intercepts; with large centers and high ICC the two specifications converge."""

    cfg = { 'lambda_local': 1.0, 'lambda_global': 1.0, 'cv': { 'n_splits': 5, 'n_repeats': 2, 'seed': 0 } }
    reg = build_registry( cfg, only = [ 'BFI-Fixed', 'BFI-Hetero' ] )

    small = simulate_dataset( n_clusters = 25, n_per_cluster = 5, nonlin = 0.0, tau = 0.3, seed = 21 )
    res   = CVRunner( small, cfg, reg ).run( verbose = False )
    r2    = { m: point_metrics( res.y_true, P )['R2'] for m, P in res.preds.items() }
    assert r2['BFI-Hetero'] > r2['BFI-Fixed'] + 0.05, r2

    large = simulate_dataset( n_clusters = 25, n_per_cluster = 80, nonlin = 0.0, tau = 0.8, seed = 22 )
    res   = CVRunner( large, cfg, reg ).run( verbose = False )
    r2    = { m: point_metrics( res.y_true, P )['R2'] for m, P in res.preds.items() }
    assert abs( r2['BFI-Hetero'] - r2['BFI-Fixed'] ) < 0.01, r2


def test_local_ridge_tuned_runs_with_tiny_centers():

    raw = simulate_dataset( n_clusters = 8, n_per_cluster = 5, seed = 2 )
    cfg = { 'cv': { 'n_splits': 5, 'n_repeats': 1, 'seed': 0 } }
    res = CVRunner( raw, cfg, build_registry( cfg, only = [ 'Ridge (local, tuned)' ] ) ).run( verbose = False )
    assert np.isfinite( res.preds['Ridge (local, tuned)'] ).all()


def test_simulation_incremental_resume_adds_only_missing_models():

    import tempfile, pathlib, pandas as pd
    from simulation import run_simulation
    base = { 'lambda_local': 1.0, 'lambda_global': 1.0 }
    sim  = { 'tau': [0.3], 'nonlin': [0.0], 'n_per_cluster': [5], 'n_clusters': 8, 'n_datasets': 3,
             'n_datasets_neural': 0, 'n_jobs': 1, 'models_neural': [], 'reference': 'BFI-Hetero' }
    with tempfile.TemporaryDirectory() as d:
        cfg1 = dict( base, simulation = dict( sim, models_linear = [ 'BFI-Linear', 'BFI-Hetero' ] ) )
        raw1, _, _ = run_simulation( cfg1, out_dir = d, verbose = False )
        part = next( ( pathlib.Path( d ) / 'parts' ).glob( '*.csv' ) )
        old  = pd.read_csv( part ).drop( columns = [ 'comm_rounds' ] )        # file saved without comm_rounds
        old.to_csv( part, index = False )
        cfg2 = dict( base, simulation = dict( sim, models_linear = [ 'BFI-Linear', 'BFI-Hetero', 'BFI-Fixed', 'DLMM' ] ) )
        raw2, summ2, pr2 = run_simulation( cfg2, out_dir = d, verbose = False )
        assert set( raw2['model'] ) == { 'BFI-Linear', 'BFI-Hetero', 'BFI-Fixed', 'DLMM' }
        assert ( raw2.groupby( 'model' )['rep'].nunique() == 3 ).all()
        a = raw1[ raw1['model'] == 'BFI-Hetero' ].sort_values( 'rep' )['R2'].values
        b = raw2[ raw2['model'] == 'BFI-Hetero' ].sort_values( 'rep' )['R2'].values
        assert np.allclose( a, b )                                             # existing rows preserved
        assert ( summ2.set_index( 'model' ).loc['BFI-Hetero', 'comm_rounds'] == 1.0 )
        assert set( pr2['model'] ) == { 'BFI-Linear', 'BFI-Fixed', 'DLMM' }


# ------------------------------------------------------------------ missing data

def test_missing_data_report_and_files():

    import tempfile, pathlib, json, pandas as pd, reporting
    from data_handler import DataManager
    rng = np.random.default_rng( 0 )
    n   = 60
    df  = pd.DataFrame( { 'cl': np.repeat( [ 'A', 'B', 'C', 'D' ], [ 20, 20, 17, 3 ] ),
                          'a' : rng.normal( size = n ), 'b': rng.integers( 0, 2, n ).astype( float ),
                          'z' : rng.normal( size = n ), 'y': rng.normal( size = n ) } )
    df.loc[ 0:4, 'a' ]            = np.nan          # 5 missing in A
    df.loc[ df.cl == 'B', 'b' ]   = np.nan          # covariate entirely missing in center B
    df.loc[ [ 45, 46 ], 'y' ]     = np.nan          # missing target (center C) -> rows dropped
    df.loc[ 47, 'cl' ]            = np.nan          # missing cluster -> row dropped
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path( d ) / 'toy.csv'
        df.to_csv( path, index = False )
        raw = DataManager( path, [ 'y' ], [ 'a' ], [ 'b' ], [ 'z' ], cluster_col = 'cl',
                           min_cluster_size = 5, name = 'toy' ).load_raw()
        s = raw.missing['summary']
        assert s['n_rows_csv'] == 60 and s['n_dropped_target_missing'] == 2
        assert s['n_dropped_cluster_missing'] == 1 and s['n_dropped_small_clusters'] == 3
        assert s['n_subjects'] == raw.n == 54 and s['n_clusters'] == 3
        bf = raw.missing['by_feature'].set_index( 'feature' )
        assert bf.loc['a', 'n_missing'] == 5 and bf.loc['b', 'n_missing'] == 20 and bf.loc['z', 'n_missing'] == 0
        assert bf.loc['b', 'n_clusters_all_missing'] == 1 and np.isclose( bf.loc['a', 'pct_missing'], 500 / 54 )
        assert s['n_complete_cases'] == 54 - 25 and s['n_cluster_feature_all_missing'] == 1
        assert np.isclose( s['pct_cells_missing'], 100 * np.isnan( raw.X ).mean() )
        bc = raw.missing['by_cluster'].set_index( 'cluster' )
        assert bc.loc['B', 'n_features_all_missing'] == 1 and bc.loc['C', 'pct_cells_missing'] == 0
        reporting.save_missing_data( raw.missing, d )
        reporting.save_missing_summary( [ raw.missing, raw.missing ], d )
        for f in [ 'missing_summary.json', 'missing_by_feature.csv', 'missing_by_cluster.csv',
                   'missing_cluster_x_feature.csv', 'missing_patterns.csv', 'Missing_Data.dat',
                   'missing_data_summary.csv', 'missing_data_summary.dat', 'missing_data_by_feature_all.csv' ]:
            assert ( pathlib.Path( d ) / f ).exists(), f
        assert len( pd.read_csv( pathlib.Path( d ) / 'missing_data_summary.csv' ) ) == 2
        json.loads( ( pathlib.Path( d ) / 'missing_summary.json' ).read_text() )


if __name__ == "__main__":

    tests = [ v for k, v in sorted( globals().items() ) if k.startswith( 'test_' ) ]
    for t in tests:
        t()
        print( f"OK  {t.__name__}" )
    print( f"\n{len(tests)} tests passed." )
