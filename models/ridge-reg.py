import h5py
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

uni_dir = "/blue/cis6930/reardons/CANCER-TRANS-2025/UNI-results"
ref_csv = "/blue/cis6930/reardons/data/TCGA2025/sequoia_reference.csv"

X = pd.read_csv("foo.csv")
X.set_index('Unnamed: 0', inplace=True)
Y = pd.read_csv(ref_csv)
Y['patient_id'] = Y['wsi_file_name'].str.split('.').str[0].str.split('-01Z').str[0]
Y.set_index('patient_id', inplace=True)
Y = Y.drop(['wsi_file_name', 'tcga_project'], axis=1)

# Make sure X and Y are aligned
common_patients = X.index.intersection(Y.index)
X = X.loc[common_patients]
Y = Y.loc[common_patients]

# Normalization (log2 transform on Y)
Y_log2 = Y.map(lambda x: np.log2(x+0.01))

# Convert feature and ground truth to numpy arrays
X_np = X.to_numpy()
Y_np = Y_log2.to_numpy()

def calculate_metrics(y_true, y_pred):
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
            # If either is constant, Pearson correlation is not defined
            return np.nan, np.sqrt(mean_squared_error(y_true, y_pred))  # Return NaN for Pearson if constant
    
    # Pearson's correlation
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return pearson_corr, rmse

def calculate_mean_p(y_true, y_pred):
    # Compute Pearson correlation for each target variable separately
    pearson_corrs = []
    for i in range(y_true.shape[1]):  # Iterate over all columns (output features)
        if np.all(y_true[:, i] == y_true[0, i]) or np.all(y_pred[:, i] == y_pred[0, i]):
            pearson_corrs.append(np.nan)  # Avoid computing Pearson for constant values
        else:
            pearson_corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
            pearson_corrs.append(pearson_corr)
    
    # Take the mean Pearson correlation across all features
    mean_pearson_corr = np.nanmean(pearson_corrs)  # Ignore NaNs
    return mean_pearson_corr

# Hyperparameter tuning for Ridge
alphas = np.logspace(-4, 4, 100)  # Search between 10^-4 and 10^4
ridge_reg = RidgeCV(alphas=alphas, store_cv_values=True)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_info = {
    'fold': [],
    'train_score': [],
    'test_score': [],
    'train_mean_pearson': [],
    'test_mean_pearson': [],
    'train_rmse': [],
    'test_rmse': [],
}

for fold, (train_index, test_index) in enumerate(k_fold.split(X)):
    X_train, X_test = X_np[train_index], X_np[test_index]
    Y_train, Y_test = Y_np[train_index], Y_np[test_index]

    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)

    # Dimensionality reduction
    pca = PCA(n_components=100)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    train_patient_ids = Y.iloc[train_index].index
    test_patient_ids = Y.iloc[test_index].index
    
    # Train RidgeCV to find the best alpha
    ridge_reg.fit(X_train, Y_train)
    best_alpha = ridge_reg.alpha_
    
    # Train Ridge with the best alpha
    ridge_final = RidgeCV(alphas=[best_alpha])
    ridge_final.fit(X_train, Y_train)
    
    # Make predictions
    train_pred = ridge_final.predict(X_train)
    test_pred = ridge_final.predict(X_test)
    
    # R2 scores
    train_score = r2_score(Y_train, train_pred)
    test_score = r2_score(Y_test, test_pred)
    
    # Calculate Pearson's correlation and RMSE for training and testing
    train_pearson, train_rmse = calculate_metrics(Y_train, train_pred)
    test_pearson, test_rmse = calculate_metrics(Y_test, test_pred)

    train_mean_p = calculate_mean_p(Y_train, train_pred)
    test_mean_p = calculate_mean_p(Y_test, test_pred)
    
    # Store fold information
    fold_info['fold'].append(fold + 1)
    fold_info['train_score'].append(train_score)
    fold_info['test_score'].append(test_score)
    fold_info['train_mean_pearson'].append(train_mean_p)
    fold_info['test_mean_pearson'].append(test_mean_p)
    np.save('results/raw-results/ridge/ridge-train_pearson_fold_{}.npy'.format(fold + 1), train_pearson)
    np.save('results/raw-results/ridge/ridge-test_pearson_fold_{}.npy'.format(fold + 1), test_pearson)
    fold_info['train_rmse'].append(train_rmse)
    fold_info['test_rmse'].append(test_rmse)

    train_predictions_df = pd.DataFrame(train_pred, index=train_patient_ids, columns=Y.columns)
    test_predictions_df = pd.DataFrame(test_pred, index=test_patient_ids, columns=Y.columns)

    train_predictions_df.to_csv(f'results/raw-results/ridge/fold_{fold + 1}_train_predictions.csv')
    test_predictions_df.to_csv(f'results/raw-results/ridge/fold_{fold + 1}_test_predictions.csv')

# Convert to DataFrame for better organization
fold_info_df = pd.DataFrame.from_dict(fold_info, orient='index')

# Save fold-wise metrics to a CSV file
fold_info_df.to_csv('results/ridge-kfold_cv_results.csv')