import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import pickle as pk

import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load and preprocess data
data = pd.read_csv(os.path.join(current_dir, 'Bengaluru_House_Data.csv'))

# Drop unnecessary columns
data.drop(columns=['area_type', 'society', 'availability'], inplace=True)

# Handle range values and different units in total_sqft
def convert_sqft_to_num(x):
    if isinstance(x, str):
        # Handle range values
        if '-' in x:
            nums = x.split('-')
            return (float(nums[0]) + float(nums[1])) / 2
        # Remove 'Sq. Meter', 'Sq. Yards', etc. and convert to float
        x = x.replace('Sq. Meter', '').replace('Sq. Yards', '').replace('Perch', '').replace('Grounds', '').strip()
        try:
            return float(x)
        except ValueError:
            return None

# Convert and clean total_sqft
data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)

# Remove rows with invalid total_sqft
data = data.dropna(subset=['total_sqft'])

# Remove outliers and invalid values
data = data[(data['total_sqft'] > 100) & (data['total_sqft'] < 10000)]

# Extract number of bedrooms from 'size' column
data['bedrooms'] = data['size'].str.extract('(\d+)').astype(float)

# Drop the original 'size' column
data.drop(columns=['size'], inplace=True)

# Handle missing values
data['bath'] = data['bath'].fillna(data['bath'].median())
data['balcony'] = data['balcony'].fillna(data['balcony'].median())
data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].median())

# Save cleaned data
data.to_csv(os.path.join(current_dir, 'cleaned_data.csv'), index=False)

# Prepare features and target
X = data[['location', 'total_sqft', 'bath', 'balcony', 'bedrooms']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the pipeline
model = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['location']),
        remainder='passthrough'
    ),
    StandardScaler(),
    LinearRegression()
)

# Fit the model
model.fit(X_train, y_train)

# Save the model
pk.dump(model, open(os.path.join(current_dir, 'House_prediction_model.pkl'), 'wb'))

# Print model score
print(f'Model Score: {model.score(X_test, y_test)}')

# Test prediction
test_input = pd.DataFrame(
    [['Electronic City Phase II', 2000.0, 3.0, 2.0, 3]], 
    columns=['location', 'total_sqft', 'bath', 'balcony', 'bedrooms']
)
print(f'Test Prediction: {model.predict(test_input)[0]}')