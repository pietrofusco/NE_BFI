#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:57:16 2026

@author: peter

Data loading, preprocessing and cross-validation splits.

  * DataManager.load_raw()  reads the CSV with no data-dependent transformation (only drops rows
                            with missing target/cluster and removes clusters that are too small).
  * Preprocessor            imputation + standardization, fitted on the training indices of each
                            fold only, so that no test-set statistics leak into training.
  * make_repeated_cv_splits repeated k-fold, stratified by cluster (each cluster is split into k parts).
  * build_fold              builds local/pooled training sets and the pooled test set for one fold,
                            with optional subsampling of the training set (for learning curves).

Preprocessing: all columns of X are centered (continuous ones also scaled to unit variance) and Y
is centered and scaled.  This way the linear model without intercept (BFI) is exactly equivalent to
a ridge with intercept on the pooled training data, and the baselines are comparable.
Means and SDs are aggregable statistics (sums and sums of squares), so in a federated setting they
can be shared without moving individual records.
"""
# %%

import numpy  as np
import pandas as pd


from dataclasses           import dataclass
from dataclasses           import field
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import StandardScaler


# %%

@dataclass
class RawData:
    """Raw data (X may contain NaN), no learned transformation."""

    X             : np.ndarray            # (N, p) float, may contain NaN
    Y             : np.ndarray            # (N, q) float, no NaN
    clusters      : np.ndarray            # (N,)   cluster labels
    feature_names : list                  # length p
    groups        : np.ndarray            # (p,)   'cont' | 'bin' | 'z'
    target_names  : list                  # length q
    name          : str = "dataset"
    info          : dict = field( default_factory = dict )
    # missing-data report (see missing_data_report); empty for simulated data
    missing       : dict = field( default_factory = dict )

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def input_dim(self):
        return self.X.shape[1]


# %%

class DataManager:

    def __init__( self,
                  file_path,
                  target_cols,
                  continuous_cols,
                  binary_cols,
                  zscore_cols        = (),
                  cluster_col        = 'cluster',
                  min_cluster_size   = 10,
                  name               = None ):

        self.file_path        = file_path
        self.target_cols      = list(target_cols)
        self.continuous_cols  = list(continuous_cols)
        self.binary_cols      = list(binary_cols)
        self.zscore_cols      = list(zscore_cols)
        self.cluster_col      = cluster_col
        self.min_cluster_size = min_cluster_size
        self.name             = name if name is not None else str(file_path)


    def load_raw(self):

        df = pd.read_csv( filepath_or_buffer = self.file_path )

        required = ( self.target_cols + [self.cluster_col] + self.continuous_cols +
                     self.binary_cols + self.zscore_cols )
        missing  = [ c for c in required if c not in df.columns ]
        if missing:

            raise KeyError( f"[{self.name}] columns not found in CSV: {missing}. "
                            f"Available: {list(df.columns)}" )

        feature_names = self.continuous_cols + self.binary_cols + self.zscore_cols

        # copy of the CSV before any exclusion (for the missing-data report)
        df_csv = df[ required ].copy()

        n_csv          = len(df)
        miss_target    = df[self.target_cols].isna().any( axis = 1 )
        miss_cluster   = df[self.cluster_col].isna()
        df = df.dropna( subset = self.target_cols + [self.cluster_col] )

        # drop clusters too small for a sensible CV
        sizes     = df[self.cluster_col].value_counts()
        too_small = sizes[sizes < self.min_cluster_size].index.tolist()
        n_before_small = len(df)
        if len(too_small) > 0:

            print(f"[{self.name}] removed {len(too_small)} clusters with "
                  f"n < {self.min_cluster_size}: {too_small}")
            df = df[ ~df[self.cluster_col].isin(too_small) ]

        exclusions = { 'n_rows_csv'                  : int(n_csv),
                       'n_dropped_target_missing'    : int(miss_target.sum()),
                       'n_dropped_cluster_missing'   : int((miss_cluster & ~miss_target).sum()),
                       'n_dropped_small_clusters'    : int(n_before_small - len(df)),
                       'small_clusters_removed'      : [ str(c) for c in too_small ],
                       'n_rows_analysed'             : int(len(df)) }
        groups        = np.array( ['cont'] * len(self.continuous_cols) +
                                  ['bin']  * len(self.binary_cols)     +
                                  ['z']    * len(self.zscore_cols) )

        X        = df[feature_names].to_numpy( dtype = float )
        Y        = df[self.target_cols].to_numpy( dtype = float )
        clusters = df[self.cluster_col].to_numpy()

        cluster_sizes = pd.Series(clusters).value_counts()
        info = { 'n_subjects'      : int(len(df)),
                 'n_clusters'      : int(cluster_sizes.shape[0]),
                 'cluster_size_min': int(cluster_sizes.min()),
                 'cluster_size_med': float(cluster_sizes.median()),
                 'cluster_size_max': int(cluster_sizes.max()),
                 'missing_X_pct'   : float(100.0 * np.isnan(X).mean()) }

        missing = missing_data_report( X, clusters, feature_names, groups,
                                       df_csv        = df_csv,
                                       target_cols   = self.target_cols,
                                       cluster_col   = self.cluster_col,
                                       exclusions    = exclusions,
                                       name          = self.name )
        info['missing_complete_cases_pct'] = missing['summary']['pct_complete_cases']

        return RawData( X = X, Y = Y, clusters = clusters,
                        feature_names = feature_names, groups = groups,
                        target_names = self.target_cols, name = self.name,
                        info = info, missing = missing )


# %% ---------------------------------------------------- missing data

def missing_data_report( X,
                         clusters,
                         feature_names,
                         groups,
                         df_csv      = None,
                         target_cols = (),
                         cluster_col = 'cluster',
                         exclusions  = None,
                         name        = 'dataset',
                         top_patterns = 20 ):
    """
    Describes missing covariate values in the analysed sample (after dropping rows without
    target/cluster and clusters that are too small) and, if df_csv is given, in the original CSV.

    Returns a dict with
       'summary'     : dict (one row of the overall summary table)
       'by_feature'  : DataFrame, one row per covariate (n and % missing, group, imputation
                       strategy, % in the original CSV, number of centers where the covariate is
                       entirely missing / has at least one missing value, min/median/max % across centers)
       'by_cluster'  : DataFrame, one row per center (n subjects, % missing cells, % subjects with at
                       least one missing value, number of covariates entirely missing in the center)
       'cluster_x_feature' : DataFrame centers x covariates with the % missing
       'patterns'    : DataFrame of the most frequent missingness patterns (1 = missing)
       'targets_csv' : DataFrame, % missing of target and cluster in the original CSV
    """
    X        = np.asarray( X, dtype = float )
    M        = np.isnan( X )
    n, p     = X.shape
    clusters = np.asarray( clusters )
    groups   = np.asarray( groups )
    strategy = { 'cont': 'median', 'bin': 'most_frequent', 'z': 'median' }

    # ---------------------------------------------- center x covariate
    cl_ser = pd.Series( clusters, name = 'cluster' )
    cxf    = pd.DataFrame( M, columns = feature_names ).groupby( cl_ser.values ).mean() * 100.0
    cxf.index.name = 'cluster'

    # ---------------------------------------------- by covariate
    by_feature = pd.DataFrame( { 'feature'                 : feature_names,
                                 'group'                   : groups,
                                 'imputation'              : [ strategy.get( g, '' ) for g in groups ],
                                 'n_missing'               : M.sum( axis = 0 ).astype( int ),
                                 'pct_missing'             : 100.0 * M.mean( axis = 0 ) if n else np.nan,
                                 'n_clusters_any_missing'  : ( cxf > 0 ).sum( axis = 0 ).values.astype( int ),
                                 'n_clusters_all_missing'  : ( cxf >= 100.0 ).sum( axis = 0 ).values.astype( int ),
                                 'pct_missing_cluster_min' : cxf.min( axis = 0 ).values,
                                 'pct_missing_cluster_med' : cxf.median( axis = 0 ).values,
                                 'pct_missing_cluster_max' : cxf.max( axis = 0 ).values } )
    if df_csv is not None and len(df_csv) > 0:
        by_feature['pct_missing_csv'] = [ 100.0 * df_csv[f].isna().mean() for f in feature_names ]
    by_feature.insert( 0, 'dataset', name )

    # ---------------------------------------------- by center
    row_any = M.any( axis = 1 )
    by_cluster = ( pd.DataFrame( { 'cluster'  : clusters,
                                   'cells'    : M.mean( axis = 1 ) * 100.0,
                                   'row_any'  : row_any.astype( float ) * 100.0 } )
                   .groupby( 'cluster' )
                   .agg( n_subjects            = ( 'cells', 'size' ),
                         pct_cells_missing     = ( 'cells', 'mean' ),
                         pct_subjects_any_missing = ( 'row_any', 'mean' ) ) )
    by_cluster['n_features_all_missing'] = ( cxf >= 100.0 ).sum( axis = 1 ).reindex( by_cluster.index ).astype( int )
    by_cluster = by_cluster.reset_index()
    by_cluster.insert( 0, 'dataset', name )

    # ---------------------------------------------- missingness patterns
    if n > 0:
        pat = pd.DataFrame( M.astype( int ), columns = feature_names )
        patterns = ( pat.value_counts().rename( 'n_subjects' ).reset_index()
                        .sort_values( 'n_subjects', ascending = False ) )
        patterns['n_features_missing'] = patterns[feature_names].sum( axis = 1 )
        patterns['pct_subjects']       = 100.0 * patterns['n_subjects'] / n
        n_patterns = int( len(patterns) )
        patterns   = patterns.head( top_patterns ).reset_index( drop = True )
    else:
        patterns, n_patterns = pd.DataFrame(), 0

    # ---------------------------------------------- target / cluster in the CSV
    targets_csv = None
    if df_csv is not None:
        cols = list( target_cols ) + [ cluster_col ]
        targets_csv = pd.DataFrame( { 'dataset'     : name,
                                      'column'      : cols,
                                      'role'        : [ 'target' ] * len(target_cols) + [ 'cluster' ],
                                      'n_missing'   : [ int( df_csv[c].isna().sum() ) for c in cols ],
                                      'pct_missing' : [ 100.0 * df_csv[c].isna().mean() if len(df_csv) else np.nan
                                                        for c in cols ] } )

    # ---------------------------------------------- summary
    feat_pct = by_feature.set_index( 'feature' )['pct_missing']
    summary  = { 'dataset'                     : name,
                 **( exclusions or {} ),
                 'n_subjects'                    : int(n),
                 'n_clusters'                    : int( len( np.unique( clusters ) ) ),
                 'n_features'                    : int(p),
                 'pct_cells_missing'             : float( 100.0 * M.mean() ) if M.size else 0.0,
                 'n_complete_cases'              : int( (~row_any).sum() ),
                 'pct_complete_cases'            : float( 100.0 * (~row_any).mean() ) if n else np.nan,
                 'n_features_with_missing'       : int( (feat_pct > 0).sum() ),
                 'features_with_missing'         : ', '.join( f"{f} ({v:.1f}%)" for f, v in
                                                            feat_pct[ feat_pct > 0 ].sort_values( ascending = False ).items() ),
                 'max_feature_pct_missing'       : float( feat_pct.max() ) if p else 0.0,
                 'max_feature'                   : str( feat_pct.idxmax() ) if p and feat_pct.max() > 0 else '',
                 'n_clusters_with_missing'       : int( ( by_cluster['pct_cells_missing'] > 0 ).sum() ),
                 'n_cluster_feature_all_missing' : int( ( cxf >= 100.0 ).values.sum() ),
                 'n_missing_patterns'            : n_patterns }
    if 'small_clusters_removed' in summary:
        summary['small_clusters_removed'] = ', '.join( summary['small_clusters_removed'] )
    for g in ( 'cont', 'bin', 'z' ):
        cols = np.where( groups == g )[0]
        summary[f'pct_cells_missing_{g}'] = float( 100.0 * M[:, cols].mean() ) if len(cols) and n else np.nan

    return { 'summary'           : summary,
             'by_feature'        : by_feature,
             'by_cluster'        : by_cluster,
             'cluster_x_feature' : cxf.reset_index(),
             'patterns'          : patterns,
             'targets_csv'       : targets_csv }


# %%

class Preprocessor:
    """Imputation + standardization, to be fitted on the training data only."""

    _STRATEGY = { 'cont': 'median', 'bin': 'most_frequent', 'z': 'median' }

    def __init__( self,
                  groups ):

        self.groups    = np.asarray(groups)
        self.imputers  = {}
        self.scaler_X  = None
        self.scaler_Y  = StandardScaler()


    def fit( self,
             X,
             Y ):

        X_imp = np.empty_like(X, dtype = float)
        for g, strategy in self._STRATEGY.items():

            cols = np.where(self.groups == g)[0]
            if len(cols) == 0:
                continue

            imp = SimpleImputer( strategy = strategy, keep_empty_features = True )
            X_imp[:, cols] = imp.fit_transform( X[:, cols] )
            self.imputers[g] = imp

        # center all columns; scale only the continuous ones to unit variance
        self.mean_X = X_imp.mean( axis = 0 )
        std         = X_imp.std( axis = 0 )
        std[std < 1e-12] = 1.0
        self.std_X  = np.where( self.groups == 'cont', std, 1.0 )

        self.scaler_Y.fit( Y )

        return self


    def transform_X( self,
                     X ):

        X_imp = np.empty_like(X, dtype = float)
        for g, imp in self.imputers.items():

            cols = np.where(self.groups == g)[0]
            X_imp[:, cols] = imp.transform( X[:, cols] )

        return ( X_imp - self.mean_X ) / self.std_X


    def transform_Y( self,
                     Y ):

        return self.scaler_Y.transform( Y )


    def inverse_Y( self,
                   Y_scaled ):

        return self.scaler_Y.inverse_transform( Y_scaled )


# %%

def make_repeated_cv_splits( clusters,
                             n_splits  = 5,
                             n_repeats = 10,
                             seed      = 0 ):
    """
    Repeated k-fold stratified by cluster: in each repetition every cluster is shuffled and spread
    over the k folds (with a random offset to balance fold sizes).  Each subject appears in the test
    set exactly once per repetition.

    Returns a list of dicts {repeat, fold, train_idx, test_idx}.
    """
    clusters = np.asarray(clusters)
    splits   = []

    for r in range(n_repeats):

        rng      = np.random.default_rng( seed + 1000 * r )
        fold_ids = np.empty( len(clusters), dtype = int )

        for c in np.unique(clusters):

            idx    = np.where(clusters == c)[0]
            idx    = rng.permutation(idx)
            offset = rng.integers( 0, n_splits )
            fold_ids[idx] = ( np.arange(len(idx)) + offset ) % n_splits

        for k in range(n_splits):

            test_idx  = np.where(fold_ids == k)[0]
            train_idx = np.where(fold_ids != k)[0]
            splits.append( { 'repeat'    : r,
                             'fold'      : k,
                             'train_idx' : train_idx,
                             'test_idx'  : test_idx } )

    return splits


# %%

def _subsample_train( train_idx,
                      clusters,
                      fraction,
                      rng,
                      min_keep = 4 ):
    """Subsample the training set within each cluster (for learning curves)."""

    if fraction >= 1.0:
        return train_idx

    kept = []
    for c in np.unique( clusters[train_idx] ):

        idx_c = train_idx[ clusters[train_idx] == c ]
        n_keep = int( max( min_keep, round( fraction * len(idx_c) ) ) )
        n_keep = min( n_keep, len(idx_c) )
        kept.append( rng.choice( idx_c, size = n_keep, replace = False ) )

    return np.sort( np.concatenate(kept) )


def build_fold( raw,
                train_idx,
                test_idx,
                train_fraction = 1.0,
                seed           = 0 ):
    """
    Builds the data structures for one fold.  Preprocessing is fitted on train only.

    fold = { 'local'        : {c: {'X_train','Y_train','n_train'}},
             'pooled_train' : {'X','Y','cluster','idx'},
             'pooled_test'  : {'X','Y','Y_raw','cluster','idx'},
             'preproc'      : Preprocessor,
             'input_dim'    : int }
    """
    rng       = np.random.default_rng(seed)
    train_idx = _subsample_train( train_idx, raw.clusters, train_fraction, rng )

    preproc = Preprocessor( raw.groups ).fit( raw.X[train_idx], raw.Y[train_idx] )

    X_tr = preproc.transform_X( raw.X[train_idx] )
    Y_tr = preproc.transform_Y( raw.Y[train_idx] )
    X_te = preproc.transform_X( raw.X[test_idx] )
    Y_te = preproc.transform_Y( raw.Y[test_idx] )
    cl_tr = raw.clusters[train_idx]
    cl_te = raw.clusters[test_idx]

    local = {}
    for c in np.unique(cl_tr):

        m = cl_tr == c
        local[c] = { 'X_train' : X_tr[m],
                     'Y_train' : Y_tr[m],
                     'n_train' : int(m.sum()) }

    unseen = set( np.unique(cl_te) ) - set( local.keys() )
    if unseen:
        raise ValueError( f"Clusters in the test set but not in the training set: {unseen}" )

    return { 'local'        : local,
             'pooled_train' : { 'X': X_tr, 'Y': Y_tr, 'cluster': cl_tr, 'idx': train_idx },
             'pooled_test'  : { 'X': X_te, 'Y': Y_te, 'Y_raw': raw.Y[test_idx],
                                'cluster': cl_te, 'idx': test_idx },
             'preproc'      : preproc,
             'input_dim'    : X_tr.shape[1] }


# %%
