import h5py
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization

# Define the file paths and read the data
uni_dir = "/blue/cis6930/reardons/CANCER-TRANS-2025/UNI-results"
ref_csv = "/blue/cis6930/reardons/data/TCGA2025/sequoia_reference.csv"

# Load the feature and label data
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

# Function to calculate the Pearson correlation and RMSE
def calculate_metrics(y_true, y_pred):
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
            # If either is constant, Pearson correlation is not defined
            return np.nan, np.sqrt(mean_squared_error(y_true, y_pred))  # Return NaN for Pearson if constant
    
    # Pearson's correlation
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return pearson_corr, rmse

# Function to calculate mean Pearson's correlation across all target variables
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

def build_mlp(input_shape, output_shape):
    model = keras.Sequential([
        layers.Dense(4096, activation='relu', input_shape=(input_shape,)),
        layers.Dense(4096, activation='relu'),
        BatchNormalization(),
        layers.Dense(4096, activation='relu'),
        BatchNormalization(),
        layers.Dense(4096, activation='relu'),
        BatchNormalization(),
        layers.Dense(4096, activation='relu'),
        BatchNormalization(),
        layers.Dense(2048, activation='relu'),
        BatchNormalization(),
        layers.Dense(2048, activation='relu'),
        BatchNormalization(),
        layers.Dense(2048, activation='relu'),
        BatchNormalization(),
        layers.Dense(2048, activation='relu'),
        BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        BatchNormalization(),
        layers.Dense(512, activation='relu'),
        BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dense(output_shape, activation='linear')  # Linear activation for regression
    ])
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.01, clipvalue=0.5), loss='mse', metrics=['mae'])
    return model

# Define k-fold cross-validation parameters
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

# Loop through each fold and train/evaluate the MLP model
for fold, (train_index, test_index) in enumerate(k_fold.split(X_np)):
    # Create train and test sets
    X_train, X_test = X_np[train_index], X_np[test_index]
    Y_train, Y_test = Y_np[train_index], Y_np[test_index]

    scale = StandardScaler().fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)
    
    train_patient_ids = Y.iloc[train_index].index
    test_patient_ids = Y.iloc[test_index].index
    
    # Initialize the MLPRegressor model
    mlp = build_mlp(input_shape=X_train.shape[1], output_shape=Y_train.shape[1])
    
    # Train the MLP model
    mlp.fit(X_train, Y_train, validation_split=0.2, epochs=200, batch_size=32, verbose=1)
    
    # Make predictions on train and test sets
    train_pred = mlp.predict(X_train)
    test_pred = mlp.predict(X_test)
    
    # Calculate R2 scores for both training and testing sets
    train_score = r2_score(Y_train, train_pred)
    test_score = r2_score(Y_test, test_pred)
    
    # Calculate Pearson's correlation and RMSE for both training and testing sets
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
    np.save('results/raw-results/mlp/mlp-train_pearson_fold_{}.npy'.format(fold + 1), train_pearson)
    np.save('results/raw-results/mlp/mlp-test_pearson_fold_{}.npy'.format(fold + 1), test_pearson)
    fold_info['train_rmse'].append(train_rmse)
    fold_info['test_rmse'].append(test_rmse)
    
    # Save fold predictions to CSV files (train and test predictions)
    train_predictions_df = pd.DataFrame(train_pred, index=train_patient_ids, columns=Y.columns)
    test_predictions_df = pd.DataFrame(test_pred, index=test_patient_ids, columns=Y.columns)

    train_predictions_df.to_csv(f'results/raw-results/mlp/fold_{fold + 1}_train_predictions.csv')
    test_predictions_df.to_csv(f'results/raw-results/mlp/fold_{fold + 1}_test_predictions.csv')

# Convert the fold results into a DataFrame for better organization
fold_info_df = pd.DataFrame.from_dict(fold_info, orient='index')

# Save the k-fold metrics to a CSV file
fold_info_df.to_csv('results/mlp-kfold_cv_results.csv')