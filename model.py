import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
#import seaborn as sns

# Assuming df is your dataframe with numeric columns, timestamp, and system_status_running

# 1. Feature Engineering - Create lag features from historical data
def create_lag_features(df, lag_columns, lag_steps=[50]):
    """
    Create lag features for selected columns.
    lag_columns: List of column names to create lags for
    lag_steps: How many timesteps to look back
    """
    df_copy = df.copy()
    
    # Sort by timestamp to ensure correct lag creation
    df_copy = df_copy.sort_values('timestamp')
    
    for col in lag_columns:
        for lag in lag_steps:
            lag_name = f'{col}_lag_{lag}'
            df_copy[lag_name] = df_copy[col].shift(lag)
    
    # Drop rows with NaN values (the initial rows where we don't have enough history)
    df_copy = df_copy.dropna(axis=1, how='all')
    df_copy = df_copy.dropna()
    
    return df_copy

# 2. Data Preparation
# Identify numeric columns excluding timestamp and target variable
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in ['timestamp', 'system_running_status']]

# Create lag features
df_with_lags = create_lag_features(df, numeric_cols)

# Prepare features (X) and target (y)
X = df_with_lags.drop(['timestamp', 'system_running_status', 'bop_plc_system_running', 'bop_plc_lp_skid_system_running'], axis=1)
y = df_with_lags['system_running_status']

str_cols = X.select_dtypes(include=['O']).columns.tolist()

for i, col in enumerate(str_cols):
    X[col] = X[col].astype(str).str.strip().map(lambda x: 1 if x.lower() == "t" else 0)

# Split data into training and testing sets - using time-based split
train_size = 0.8
split_idx = int(len(X) * train_size)


X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# drop problematic columns that seem to be perfectly correlated with system status prediction target
X=X.drop(['bop_plc_xv2101_zso'], axis=1)

# Train GBDT model
# 3. Build and train the model
# Create a pipeline with scaling and GBDT
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gbdt', GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

# Train the model
pipeline.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]  # Probability of class 1

# Print evaluation metrics
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
plt.subplot(1, 2, 2)
feature_importance = pipeline.named_steps['gbdt'].feature_importances_
sorted_idx = feature_importance.argsort()
plt.barh(np.array(X.columns)[sorted_idx][-10:], feature_importance[sorted_idx][-10:])
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.show()

# 6. Function to make predictions on new data
def predict_system_status(new_data, model, lag_columns, lag_steps=[1, 3, 6, 12, 24]):
    """
    Prepare new data and make predictions
    new_data: DataFrame with the same columns as training data
    model: Trained model pipeline
    """
    # Create lag features
    data_with_lags = create_lag_features(new_data, lag_columns, lag_steps)
    
    # Prepare features
    X_new = data_with_lags.drop(['timestamp', 'system_running_status'], axis=1)
    
    # Make predictions
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]
    
    # Return results
    result = data_with_lags[['timestamp']].copy()
    result['predicted_status'] = predictions
    result['probability'] = probabilities
    
    return result