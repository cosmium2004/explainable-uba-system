import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load your training data (adjust path and columns as needed)
data = pd.read_csv('data/raw/synthetic_cloud_logs.csv')

# Example: select numeric features for scaling
numeric_features = ['feature1', 'feature2', 'feature3']  # Replace with your actual feature names
X_train = data[numeric_features].values

# Create and fit the pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler())
])
preprocessor.fit(X_train)

# Save the fitted pipeline
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print('Preprocessing pipeline fitted and saved to models/preprocessor.pkl')