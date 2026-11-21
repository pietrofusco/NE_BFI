#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_missing_sensitivity.py - missing-data diagnostics and sensitivity analyses for ChildDev.

Usage:
    python run_missing_sensitivity.py                      # non-neural models (cfg['models'])
    python run_missing_sensitivity.py --neural             # adds the neural variants (requires torch; slow)
    python run_missing_sensitivity.py --data PATH.csv      # path of the CSV (default: the one in main.py)

Output in results/ChildDev/missing_sensitivity/:
    mcar_outcome_center.csv, mcar_smd.csv           mechanism diagnostics
    sensitivity_summary.csv, sensitivity_paired.csv  R2 and dR2 vs BFI-Hetero per scenario x model
    sensitivity_table.csv                            wide model x scenario table (supplementary)
    Missing_Sensitivity.dat                          human-readable version
"""
import argparse
import pathlib

import main
import sensitivity_missing as sm
from data_handler import DataManager
from evaluation   import DEFAULT_NEURAL_VARIANTS


def main_cli():

    ap = argparse.ArgumentParser()
    ap.add_argument( '--data',   default = None )
    ap.add_argument( '--neural', action = 'store_true' )
    ap.add_argument( '--out',    default = None )
    args = ap.parse_args()

    cfg = main.default_config()
    ds  = next( d for d in cfg['datasets'] if d['name'] == 'ChildDev' )
    raw = DataManager( file_path        = args.data or ds['data_path'],
                       target_cols      = ds['targets'],
                       continuous_cols  = ds['continuous_covariates'],
                       binary_cols      = ds.get( 'binary_covariates', [] ),
                       zscore_cols      = ds.get( 'zscore_covariates', [] ),
                       cluster_col      = ds.get( 'cluster_col', 'cluster' ),
                       min_cluster_size = ds.get( 'min_cluster_size', 10 ),
                       name             = 'ChildDev' ).load_raw()
    out = pathlib.Path( args.out or pathlib.Path( cfg.get( 'save_dir', 'results' ) ) / 'ChildDev' / 'missing_sensitivity' )
    out.mkdir( parents = True, exist_ok = True )

    # ---- mechanism diagnostics
    composites = [ 'cogcomp', 'langcomp', 'motcomp' ]
    oc, smd = sm.mcar_diagnostics( raw, subsets = { 'developmental composites': composites,
                                                    'maternal / household covariates': [ 'matage', 'matedc', 'hhh_educ' ] } )
    oc.to_csv( out / 'mcar_outcome_center.csv', index = False )
    smd.to_csv( out / 'mcar_smd.csv', index = False )

    # ---- sensitivity analysis
    models = list( cfg['models'] )
    if args.neural:
        models += list( cfg.get( 'neural_variants', DEFAULT_NEURAL_VARIANTS ).keys() )
    summ, paired = sm.run_missing_sensitivity( raw, cfg, models = models,
                                               scenarios = sm.default_scenarios( delta_features = composites ) )
    summ.to_csv( out / 'sensitivity_summary.csv', index = False )
    paired.to_csv( out / 'sensitivity_paired.csv', index = False )
    tab = sm.sensitivity_table( paired )
    tab.to_csv( out / 'sensitivity_table.csv' )

    with open( out / 'Missing_Sensitivity.dat', 'w' ) as fid:
        fid.write( "ChildDev - missing covariates: mechanism diagnostics and sensitivity analyses\n\n" )
        fid.write( "--- Children with vs without missing values (outcome: Welch; center: permutation chi-square) ---\n" )
        fid.write( oc.to_string( index = False, float_format = '%.3f' ) + "\n\n" )
        fid.write( "--- Standardized mean differences of fully observed covariates (incomplete - complete) ---\n" )
        fid.write( smd.pivot( index = 'covariate', columns = 'incomplete_on', values = 'smd' )
                      .to_string( float_format = '%.3f' ) + "\n\n" )
        fid.write( "--- R2 of the reference model (BFI-Hetero) by scenario ---\n" )
        r = summ[ summ['Model'] == 'BFI-Hetero' ][ [ 'scenario', 'n', 'R2', 'R2_lo', 'R2_hi', 'CalibSlope' ] ]
        fid.write( r.to_string( index = False, float_format = '%.3f' ) + "\n\n" )
        fid.write( "--- dR2 vs BFI-Hetero [95% paired cluster-bootstrap CI] ---\n" )
        fid.write( tab.to_string() + "\n" )

    print( f"Output -> {out}" )


if __name__ == "__main__":

    main_cli()
