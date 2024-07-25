import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')

# Data Overview
print("First few rows of the dataset:")
print(data.head())
print("\nDataset Summary:")
print(data.info())
print("\nDescriptive Statistics:")
print(data.describe())

# Feature Engineering
current_year = 2024
data['age_of_car'] = current_year - data['year']
data.drop(columns=['year'], inplace=True)

# Encode categorical features
data_encoded = pd.get_dummies(data, columns=["fuel", "seller_type", "transmission", "name", "owner"])

# Handle missing values for numeric columns only
numeric_columns = data_encoded.select_dtypes(include=[np.number]).columns
data_encoded[numeric_columns] = data_encoded[numeric_columns].fillna(data_encoded[numeric_columns].mean())

# Define features and target variable
features = data_encoded.drop(columns=['selling_price'])
target = data_encoded['selling_price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize models
random_forest_regressor = RandomForestRegressor(random_state=42)
gradient_boosting_regressor = GradientBoostingRegressor(random_state=42)

# Train the models
random_forest_regressor.fit(X_train, y_train)
gradient_boosting_regressor.fit(X_train, y_train)

# Predictions
rf_price_predictions = random_forest_regressor.predict(X_test)
gb_price_predictions = gradient_boosting_regressor.predict(X_test)

# Evaluation Metrics
rf_mean_absolute_error = mean_absolute_error(y_test, rf_price_predictions)
rf_root_mean_squared_error = mean_squared_error(y_test, rf_price_predictions, squared=False)
gb_mean_absolute_error = mean_absolute_error(y_test, gb_price_predictions)
gb_root_mean_squared_error = mean_squared_error(y_test, gb_price_predictions, squared=False)

# Cross-validation for model validation
rf_cross_val_rmse = cross_val_score(random_forest_regressor, features, target, cv=5, scoring='neg_root_mean_squared_error')
rf_cross_val_rmse_mean = np.mean(np.abs(rf_cross_val_rmse))
rf_cross_val_rmse_std = np.std(np.abs(rf_cross_val_rmse))

gb_cross_val_rmse = cross_val_score(gradient_boosting_regressor, features, target, cv=5, scoring='neg_root_mean_squared_error')
gb_cross_val_rmse_mean = np.mean(np.abs(gb_cross_val_rmse))
gb_cross_val_rmse_std = np.std(np.abs(gb_cross_val_rmse))

# Output Results
print(f"Random Forest - Mean Absolute Error (MAE): {rf_mean_absolute_error:.2f}")
print(f"Random Forest - Root Mean Squared Error (RMSE): {rf_root_mean_squared_error:.2f}")
print(f"Random Forest - Cross-validation RMSE: {rf_cross_val_rmse_mean:.2f} +/- {rf_cross_val_rmse_std:.2f}")

print(f"Gradient Boosting - Mean Absolute Error (MAE): {gb_mean_absolute_error:.2f}")
print(f"Gradient Boosting - Root Mean Squared Error (RMSE): {gb_root_mean_squared_error:.2f}")
print(f"Gradient Boosting - Cross-validation RMSE: {gb_cross_val_rmse_mean:.2f} +/- {gb_cross_val_rmse_std:.2f}")
