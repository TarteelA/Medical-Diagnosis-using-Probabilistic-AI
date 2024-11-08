import os
import pandas as pd
from sklearn.model_selection import KFold

# Load your dataset
data = pd.read_csv("./data/dementia_data-MRI-features.csv")

# Define your features (X) and target variable (y)
X = data.drop(columns=['CDR'])
y = data['CDR']

# Ensure the 'data' folder exists
os.makedirs("data", exist_ok=True)

# Set up KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and save each fold to CSV in the 'data' folder
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Combine X and y for training and testing sets
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Save to CSV files in the 'data' folder
    train_data.to_csv(f'data/train_fold_{fold + 1}.csv', index=False)
    test_data.to_csv(f'data/test_fold_{fold + 1}.csv', index=False)
    
    print(f"Fold {fold + 1} saved as data/train_fold_{fold + 1}.csv and data/test_fold_{fold + 1}.csv")
