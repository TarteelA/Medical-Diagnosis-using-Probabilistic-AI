#Import Libraries
import os
import pandas as pd
from sklearn.model_selection import KFold

#Load Dementia Dataset
Data = pd.read_csv("./Data/Dementia/dementia_data-MRI-features.csv")

#Define Features (X) And Target Variable (Y)
X = Data.drop(columns=['CDR'])
Y = Data['CDR']

#Ensure Dementia Folder Is Present
os.makedirs("Data/Dementia/", exist_ok=True)

#Set Up KFold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Perform Cross-Validation And Save Each Fold To CSV In Dementia Folder
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    #Combine X And Y For Training And Testing Sets
    train_data = pd.concat([X_train, Y_train], axis=1)
    test_data = pd.concat([X_test, Y_test], axis=1)
    
    #Save To CSV Files In Dementia Folder
    train_data.to_csv(f'data/train_fold_{fold + 1}.csv', index=False)
    test_data.to_csv(f'data/test_fold_{fold + 1}.csv', index=False)
    
    print(f"Fold {fold + 1} saved as data/train_fold_{fold + 1}.csv and data/test_fold_{fold + 1}.csv")