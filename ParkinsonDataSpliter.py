#Import Libraries
import os
import pandas as pd
from sklearn.model_selection import KFold

#Load Parkinson Dataset
Data = pd.read_csv("Data/Parkinson/parkinsons_data-VOICE-features.csv")

#Define Parkinson Features (X) And Target Variable (Y)
X = Data.drop(columns = ['status'])
Y = Data['status']

#Ensure Parkinson Folder Is Present
os.makedirs("Data/Parkinson/", exist_ok = True)

#Set Up KFold Cross-Validation
KF = KFold(n_splits = 5, shuffle = True, random_state = 42)

#Perform Cross-Validation
for fold, (train_index, test_index) in enumerate(KF.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    #Combine X And Y For Training And Testing Sets
    Train_Data = pd.concat([X_train, Y_train], axis = 1)
    Test_Data = pd.concat([X_test, Y_test], axis = 1)
    
    #Save Each Fold To CSV In Parkinson Folder
    Train_Data.to_csv(f'Data/Parkinson/train_fold_{fold + 1}.csv', index = False)
    Test_Data.to_csv(f'Data/Parkinson/test_fold_{fold + 1}.csv', index = False)