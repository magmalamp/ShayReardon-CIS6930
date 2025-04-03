import h5py
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from scipy.stats import pearsonr
from scipy.stats import uniform, randint
import cupy as cp

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

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return mean_pearson_corr, rmse

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

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# param_dist = {
#     'learning_rate': uniform(0.01, 0.3),  # Randomize from 0.01 to 0.3
#     'max_depth': randint(3, 10),  # Randomize between 3 and 10
#     'n_estimators': randint(50, 200),  # Randomize between 50 and 200
#     'subsample': uniform(0.5, 0.5),  # Randomize between 0.5 and 1.0
#     'colsample_bytree': uniform(0.5, 0.5),  # Randomize between 0.5 and 1.0
# }

print("making model")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                             n_estimators=200, 
                             max_depth=4,
                             tree_method='hist',
                             device='cuda',
                             learning_rate=0.01,
                             colsample_bytree= 0.5,
                             gamma=0.20)
print("model has been made")

# random_search = RandomizedSearchCV(
#     xgb_model, 
#     param_distributions=param_dist, 
#     n_iter=5,  # Number of random combinations to try
#     cv=2,  # 5-fold cross-validation
#     scoring='neg_mean_squared_error',  # Use RMSE for evaluation
#     verbose=1,  # Print progress
#     random_state=42,
#     n_jobs=-1  # Use all available cores
# )

# print("Starting random search")
# random_search.fit(X_np, Y_np)
# best_model = random_search.best_estimator_  # Get the best model based on cross-validation

fold_info = {
    'fold': [],
    'train_score': [],
    'test_score': [],
    'train_mean_pearson': [],
    'test_mean_pearson': [],
    'train_rmse': [],
    'test_rmse': [],
    'train_predictions': [],
    'test_predictions': []
}

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 4, 6],
#     'learning_rate': [0.01, 0.1],
#     'colsample_bytree': [0.5, 0.7],
#     'gamma': [0.1, 0.2]
# }

for fold, (train_index, test_index) in enumerate(k_fold.split(X)):
    print("\nIn fold ", fold)
    X_train, X_test = X_np[train_index], X_np[test_index]
    Y_train, Y_test = Y_np[train_index], Y_np[test_index]

    # Further split train into train (80%) and validation (20%)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)
    X_val = scale.transform(X_val)

    X_train = cp.array(X_train)
    X_test = cp.array(X_test)
    X_val = cp.array(X_val)

    train_patient_ids = Y.iloc[train_index].index
    test_patient_ids = Y.iloc[test_index].index
    
    # xgb_model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', device='cuda')
    # grid_search = GridSearchCV(xgb_model, 
    #                            param_grid, 
    #                            scoring='neg_mean_absolute_error', 
    #                            cv=1, verbose=1, n_jobs=-1)

    print("Performing Fit...")
    xgb_model.fit(X_train, Y_train)

    # Select the best model based on validation MAE
    best_model = xgb_model.best_estimator_
    print(f"Best hyperparameters: {xgb_model.best_params_}")

    # Make predictions
    print("Making predictions...")
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    
    # R2 scores
    print("Calculating R_2...")
    train_score = r2_score(Y_train, train_pred)
    test_score = r2_score(Y_test, test_pred)
    
    # Calculate Pearson's correlation and RMSE for training and testing
    print("Calculating other metrics...")
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
    np.save('results/raw-results/xgboost/xgboost-train_pearson_fold_{}.npy'.format(fold + 1), train_pearson)
    np.save('results/raw-results/xgboost/xgboost-test_pearson_fold_{}.npy'.format(fold + 1), test_pearson)
    fold_info['train_rmse'].append(train_rmse)
    fold_info['test_rmse'].append(test_rmse)

    train_predictions_df = pd.DataFrame(train_pred, index=train_patient_ids, columns=Y.columns)
    test_predictions_df = pd.DataFrame(test_pred, index=test_patient_ids, columns=Y.columns)

    train_predictions_df.to_csv(f'results/raw-results/xgboost/fold_{fold + 1}_train_predictions.csv')
    test_predictions_df.to_csv(f'results/raw-results/xgboost/fold_{fold + 1}_test_predictions.csv')

# Convert to DataFrame for better organization
fold_info_df = pd.DataFrame.from_dict(fold_info, orient='index')

# Save fold-wise metrics to a CSV file
fold_info_df.to_csv('results/xgboost-kfold_cv_results.csv')