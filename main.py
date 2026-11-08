#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:04:50 2026

@author: peter

Pipeline for each dataset: repeated CV -> bootstrap CIs + paired comparisons -> learning curves
-> importance stability; then the simulation study.  Output goes to config['save_dir'].
A missing-data report is written for each dataset (Missing_Data.dat + missing_*.csv/json), plus an
overall summary across datasets in save_dir/missing_data_summary.{csv,dat} and
missing_data_by_feature_all.csv.

CONFIG
  - datasets   : list of datasets (ChildDev + Nurses from the R package 'BFI', 25 hospitals,
                 ~40 subjects per hospital, Gaussian outcome 'stress').
                 Each dataset can have 'overrides' (e.g. a smaller latent_dim, since Nurses has
                 only 5 covariates).
  - cv         : repeated k-fold stratified by cluster + bootstrap for the CIs
  - sigma2_mode: 'fixed' | 'local' | 'global' (federated estimate of sigma^2)
  - neural_variants: neural models to compare (add an ablation with 'neural':{'beta_pred':0.0}
                 for the unsupervised autoencoder)
  - hetero     : prior of the heterogeneous BFI (local center intercepts; sigma^2, tau^2 by EM)
  - federated  : FedProx settings for the neural extractor (default for all main variants)
  - experiments: switch CV, learning curves and simulation on/off

Hyperparameters are fixed a priori (no tuning on the test data).  For proper tuning, use an inner
CV on the training part of each fold only.
"""


# %%

import json
import pathlib
import time
import reporting


from data_handler import DataManager
from evaluation   import CVRunner
from evaluation   import build_registry
from evaluation   import bootstrap_all
from evaluation   import summarize
from evaluation   import paired_comparisons
from evaluation   import run_learning_curve
from evaluation   import variance_components
from evaluation   import communication_summary
from simulation   import run_simulation
from simulation   import summarize_simulation


# %%

def software_versions():
    """Exact package versions (for the Software section of the paper)."""

    import platform
    from importlib import metadata
    out = { 'python': platform.python_version() }
    for pkg in ( 'numpy', 'scipy', 'pandas', 'scikit-learn', 'statsmodels', 'torch', 'joblib', 'matplotlib' ):
        try:
            out[pkg] = metadata.version( pkg )
        except metadata.PackageNotFoundError:
            out[pkg] = None

    return out


# %%

class Orchestrator:

    def __init__( self,
                  config ):

        self.config   = config
        self.save_dir = pathlib.Path( config.get( 'save_dir', 'results' ) )
        self.missing_reports = []


    def _dataset_cfg( self,
                      ds ):
        """Global config + dataset-specific overrides (e.g. latent_dim)."""

        cfg = dict( self.config )
        cfg.update( ds.get( 'overrides', {} ) )

        return cfg


    def run_dataset( self,
                     dataset ):

        name = dataset['name']
        cfg  = self._dataset_cfg( dataset )
        out  = self.save_dir / name
        out.mkdir( parents = True, exist_ok = True )
        exp  = cfg.get( 'experiments', {} )
        cv   = cfg.get( 'cv', {} )

        print( f"\n=== DATASET: {name} ===" )
        raw = DataManager( file_path        = dataset['data_path'],
                           target_cols      = dataset['targets'],
                           continuous_cols  = dataset['continuous_covariates'],
                           binary_cols      = dataset.get( 'binary_covariates', [] ),
                           zscore_cols      = dataset.get( 'zscore_covariates', [] ),
                           cluster_col      = dataset.get( 'cluster_col', 'cluster' ),
                           min_cluster_size = dataset.get( 'min_cluster_size', 10 ),
                           name             = name ).load_raw()


        print( f"  {raw.info}" )
        with open( out / 'dataset_info.json', 'w' ) as fid:

            json.dump( raw.info, fid, indent = 2 )

        # missing-data report for this dataset; also collected for the overall summary
        reporting.save_missing_data( raw.missing, out )
        self.missing_reports.append( raw.missing )
        ms = raw.missing['summary']
        print( f"  missing: {ms['pct_cells_missing']:.2f}% of covariate cells, "
               f"complete cases {ms['pct_complete_cases']:.1f}%, "
               f"features with missing: {ms['features_with_missing'] or 'none'}" )


        registry  = build_registry( cfg )
        reference = cv.get( 'reference_model', 'BFI-Hetero' )


        # ---------------------------------------------------- 1-2) repeated CV + CIs
        if exp.get( 'cv', True ):

            print( "--> REPEATED CV" )
            t0      = time.time()
            results = CVRunner( raw, cfg, registry ).run()
            boot    = bootstrap_all( results, n_boot = cv.get( 'n_boot', 2000 ),
                                     unit = cv.get( 'boot_unit', 'cluster' ), seed = cv.get( 'seed', 0 ) )
            summary = summarize( results, boot, ci = cv.get( 'ci', 0.95 ) )
            paired  = paired_comparisons( results, boot, reference, ci = cv.get( 'ci', 0.95 ) )

            fallbacks = sum( e.get( 'fallback', 0.0 ) for e in results.extras.get( 'Mixed-effects LMM', [] ) )
            sm_noconv = sum( 1.0 - e.get( 'sm_converged', 1.0 ) for e in results.extras.get( 'Mixed-effects LMM', [] ) )
            em_noconv = sum( 1.0 - e.get( 'em_converged', 1.0 ) for e in results.extras.get( reference, [] ) )
            vc        = variance_components( results )
            vc.to_csv( out / 'variance_components.csv', index = False )
            vc_line   = ''
            if ( vc['Model'] == reference ).any():
                r = vc.set_index( 'Model' ).loc[reference]
                vc_line = ( f"\nVariance components ({reference}, standardized y, mean over folds): "
                            f"tau2 = {r.tau2:.3f}, sigma2 = {r.sigma2:.3f}, "
                            f"ICC = {r.ICC:.3f} [{r.ICC_lo:.3f}, {r.ICC_hi:.3f}]" )
            header = ( f"Dataset: {name} | {raw.info}\n"
                       f"CV: {results.meta['n_splits']}-fold x {results.meta['n_repeats']} repeats, "
                       f"bootstrap unit = '{cv.get( 'boot_unit', 'cluster' )}', reference = {reference}\n"
                       f"LMM fallback -> pooled ridge: {int(fallbacks)} folds | "
                       f"LMM not converged (statsmodels): {int(sm_noconv)} folds | "
                       f"{reference} EM at iteration cap: {int(em_noconv)} folds" + vc_line )
            print( vc_line.strip() )
            comm = communication_summary( results )
            reporting.save_tables( summary, paired, out, header = header, communication = comm )
            reporting.plot_forest( summary, out, title = name )
            reporting.plot_paired( paired, out, title = name )
            reporting.plot_scatter( results, out )

            for m in results.extras:

                if any( 'importance' in e for e in results.extras[m] ):

                    tag = ''.join( ch if ch.isalnum() else '_' for ch in m )
                    reporting.plot_importance_stability( results, m, raw.feature_names,
                                                         out / f'importance_{tag}' )

            print( summary[['Model', 'R2', 'R2_lo', 'R2_hi', 'RMSE']].to_string( index = False, float_format = '%.3f' ) )
            print( f"  ({time.time() - t0:.0f}s)" )

        # ---------------------------------------------------------- 3) learning curves
        if exp.get( 'learning_curve', True ):

            print( "--> LEARNING CURVES" )
            lc_sum, lc_diff = run_learning_curve( raw, cfg, registry, reference = reference )
            lc_sum.to_csv( out / 'learning_curve_summary.csv', index = False )
            lc_diff.to_csv( out / 'learning_curve_paired.csv', index = False )
            reporting.plot_learning_curves( lc_sum, lc_diff, out, title = name )


    def run_simulation_study( self ):

        print( "\n=== SIMULATION STUDY ===" )
        out = self.save_dir / 'simulation'
        out.mkdir( parents = True, exist_ok = True )
        raw_df, summ, diff = run_simulation( self.config, out_dir = out )
        raw_df.to_csv( out / 'simulation_raw.csv', index = False )
        summ.to_csv( out / 'simulation_summary.csv', index = False )
        diff.to_csv( out / 'simulation_paired.csv', index = False )

        # absolute R2 on the replicates shared by all models only (those including the neural model)
        sim    = self.config.get( 'simulation', {} )
        n_nn   = min( sim.get( 'n_datasets_neural', 50 ), sim.get( 'n_datasets', 200 ) )
        common = raw_df[ raw_df['rep'] < n_nn ] if n_nn > 0 else raw_df
        summ_c, _ = summarize_simulation( common, sim.get( 'reference', 'BFI-Hetero' ) )
        summ_c.to_csv( out / 'simulation_summary_common_reps.csv', index = False )
        reporting.plot_simulation( summ, diff, out, summary_common = summ_c )
        reporting.save_simulation_tables( diff, out )

        return summ, diff


    def run( self ):

        exp = self.config.get( 'experiments', {} )
        self.save_dir.mkdir( parents = True, exist_ok = True )
        with open( self.save_dir / 'software_versions.json', 'w' ) as fid:
            json.dump( software_versions(), fid, indent = 2 )
        for dataset in ( self.config['datasets'] if exp.get( 'datasets', True ) else [] ):

            self.run_dataset( dataset = dataset )

        # missing-data summary across all datasets
        if self.missing_reports:
            reporting.save_missing_summary( self.missing_reports, self.save_dir )
            print( f"\nMissing-data summary -> {self.save_dir / 'missing_data_summary.csv'}" )

        if exp.get( 'simulation', True ):

            self.run_simulation_study()

        print( "\nPipeline completed." )


# %%

if __name__ == "__main__":

                # DATASETS
    config = { 'datasets' : [ { 'name'                  : 'ChildDev',
                                'data_path'             : "../Datasets/Child_development_data280623_ENCODED.csv",
                                'targets'               : [ 'totalfci' ],
                                'continuous_covariates' : [ 'matage', 'matedc', 'hhh_educ', 'povertscore',
                                                            'cogcomp', 'langcomp', 'motcomp' ],
                                'binary_covariates'     : [ 'childsex', 'ex_bf', 'arm' ],
                                'zscore_covariates'     : [ 'underweight', 'stunting', 'wasted' ],
                                'cluster_col'           : 'cluster',
                                'min_cluster_size'      : 5 },

                              { 'name'                  : 'Nurses',
                                'data_path'             : '../Datasets/nurses.csv',   # from the R package BFI (MIT)
                                'targets'               : [ 'stress' ],
                                'continuous_covariates' : [ 'age', 'experien', 'hospsize' ],
                                'binary_covariates'     : [ 'gender', 'wardtype' ],
                                'zscore_covariates'     : [],
                                'cluster_col'           : 'hospital',
                                'min_cluster_size'      : 5,
                                'overrides'             : { 'latent_dim': 3 } } ],

                #  MODELS
               'latent_dim'      : 8,
               'lambda_local'    : 1.0,
               'lambda_global'   : 1.0,
               'sigma2_mode'     : 'global',
               'sigma2_bounds'   : (0.02, 2.0),
               'ridge_alpha'     : 1.0,
               'mlp_alpha'       : 1.0,
               'mlp_hidden'      : (32,),
               'neural'          : { 'epochs'      : 300,
                                     'lr'          : 1e-3,
                                     'alpha_recon' : 0.1,
                                     'beta_pred'   : 1.0,
                                     'hidden_dim'  : 16 },
               'hetero'          : { 'lambda_s' : 1.0,
                                     'lambda_c' : 1.0,
                                     'estimate' : True,
                                     'max_iter' : 200,
                                     'tol'      : 1e-8 },
               'fixed_intercepts': { 'lambda_int'      : 1e-3,
                                     'estimate_sigma2' : True },
               'dlmm'            : { 'reml' : True },
               'fed_em'          : { 'max_rounds' : 20,
                                     'tol'        : 1e-6 },
               # models run on the real datasets (neural variants are added automatically)
               'models'          : [ 'Ridge (local)', 'Ridge (local, tuned)', 'Ridge (pooled)',
                                     'MLP (local)', 'MLP (pooled)', 'Mixed-effects LMM',
                                     'BFI-Linear', 'BFI-Fixed', 'DLMM', 'Federated EM-LMM (iterative)',
                                     'BFI-Hetero' ],
               # All main variants are federated (FedProx).  The centralized one is a
               # non-federated reference, not an upper bound.
               'neural_variants' : { 'Neural-BFI-Hetero [X|Z] FedProx'             : { 'skip'      : True,
                                                                                       'federated' : True,
                                                                                       'hetero'    : True },
                                     'Neural-BFI-Hetero [Z] FedProx'               : { 'skip'      : False,
                                                                                       'federated' : True,
                                                                                       'hetero'    : True },
                                     'Neural-BFI-Hetero [X|Z] FedProx (unsup. AE)' : { 'skip'      : True,
                                                                                       'federated' : True,
                                                                                       'hetero'    : True,
                                                                                       'neural'    : { 'beta_pred'   : 0.0,
                                                                                                       'alpha_recon' : 1.0 } },
                                     'Neural-BFI-Hetero [X|Z] centralized'         : { 'skip'      : True,
                                                                                       'federated' : False,
                                                                                       'hetero'    : True } },
               'federated'               : { 'rounds'       : 20,
                                             'local_epochs' : 15,
                                             'mu_prox'      : 0.01 },
               'early_stopping'          : { 'enabled'      : True,
                                             'val_fraction' : 0.2,
                                             'patience'     : 3 },
               'compute_importance'      : False,      # appendix only: turn on if needed


        # EVALUATION
        'cv' : { 'n_splits'        : 5,
                 'n_repeats'       : 10,
                 'seed'            : 0,
                 'n_boot'          : 2000,
                 'ci'              : 0.95,
                 'boot_unit'       : 'cluster',
                 'reference_model' : 'BFI-Hetero' },

        'learning_curve' : { 'fractions' : [0.25, 0.5, 0.75, 1.0],
                             'n_repeats' : 3,
                             'n_boot'    : 500 },
        'simulation' : { 'tau'               : [0.3, 0.8],
                         'nonlin'            : [0.0, 0.3, 0.6],
                         'n_per_cluster'     : [5, 10, 20, 50, 100],
                         'n_clusters'        : 25,
                         'n_datasets'        : 200,
                         'n_datasets_neural' : 50,
                         'n_cv_repeats'      : 1,
                         'slope_het'         : 0.2,
                         'covariate_shift'   : 0.3,
                         'snr'               : 1.0,
                         'latent_dim'        : 8,
                         'models_linear'     : [ 'Ridge (local)', 'Ridge (pooled)', 'Mixed-effects LMM',
                                                 'BFI-Linear', 'BFI-Hetero', 'BFI-Fixed', 'DLMM',
                                                 'Federated EM-LMM (iterative)', 'Ridge (local, tuned)' ],
                         'models_neural'     : [ 'Neural-BFI-Hetero [X|Z] FedProx' ],
                         'reference'         : 'BFI-Hetero',
                         'n_jobs'            : -1 },

        'experiments' : { 'datasets'       : True,
                          'cv'             : True,
                          'learning_curve' : True,
                          'simulation'     : True },

        'save_dir' : 'results'
    }


# %%

    orchestrator = Orchestrator( config = config )

    orchestrator.run()


# %%
