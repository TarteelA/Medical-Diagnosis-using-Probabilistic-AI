#Edited By Tarteel Alkaraan (25847208)
#Updated On: 07 November 2024

#Import Libraries
import sys
import math
import time
import random
import numpy as np
import os.path
from sklearn import metrics

import BayesNetUtil as bnu
from DataReader import CSV_DataReader
from BayesNetInference import BayesNetInference

#Declare Model Evaluator Class
class ModelEvaluator(BayesNetInference):
    verbose = False 
    inference_time = None

    def __init__(self, configfile_name, datafile_test):
        if os.path.isfile(configfile_name):
            #Loads Bayesian Network Stored In ConfigFile_Name
            super().__init__(None, configfile_name, None, None)
            self.inference_time = time.time()

        #Reads Test Data Using Code From DataReader
        self.csv = CSV_DataReader(datafile_test)
        
        #Apply Discretization Checks
        self.discretize_target_variable()

        #Generates Performance Results From Above Predictions
        self.inference_time = time.time()
        true, pred, prob = self.get_true_and_predicted_targets()
        self.inference_time = time.time() - self.inference_time
        self.compute_performance(true, pred, prob)
    
    def discretize_target_variable(self):
        # Define the discretization logic (for example, using quantiles or thresholds)
        for i, data_point in enumerate(self.csv.rv_all_values):
            asf_value = float(data_point[-1])
            if asf_value < 1.0:
                self.csv.rv_all_values[i][-1] = "Low"
            elif 1.0 <= asf_value < 1.5:
                self.csv.rv_all_values[i][-1] = "Medium"
            else:
                self.csv.rv_all_values[i][-1] = "High"

    def get_true_and_predicted_targets(self):
        print("\nPERFORMING probabilistic inference on test data...")
        
        Y_true = []
        Y_pred = []
        Y_prob = []

        # Define threshold sets
        threshold_one = set(["High"])
        zero_values = set(["Low", "Medium"])  # Adjust according to discretization

        # Loop through data points to categorize targets
        for i in range(len(self.csv.rv_all_values)):
            data_point = self.csv.rv_all_values[i]
            target_value = data_point[len(self.csv.rand_vars) - 1]

            # Classify based on target_value
            if target_value in threshold_one:
                Y_true.append(1)
            elif target_value in zero_values:
                Y_true.append(0)
            else:
                # Handle other cases or log unknown values for debugging
                print(f"Unknown target value: {target_value}")
            
            # Probabilistic prediction logic placeholder
            Y_pred.append(1 if target_value in threshold_one else 0)
            Y_prob.append(float(target_value) if target_value in ["0", "1"] else 0.5)

        return Y_true, Y_pred, Y_prob

    # returns a probability distribution using Inference By Enumeration
    def get_predictions_from_BayesNet(self, data_point, nbc):
        # forms a probabilistic query based on the predictor variable,
        # the evidence (non-predictor variables), and the values of
        # the current data point (test instance) given as argument
        evidence = ""
        for var_index in range(0, len(self.csv.rand_vars)-1):
            evidence += "," if len(evidence)>0 else ""
            evidence += self.csv.rand_vars[var_index]+'='+str(data_point[var_index])
        prob_query = "P(%s|%s)" % (self.csv.predictor_variable, evidence)
        self.query = bnu.tokenise_query(prob_query, False)

        # sends query to BayesNetInference and get probability distribution
        self.prob_dist = self.enumeration_ask()
        normalised_dist = bnu.normalise(self.prob_dist)
        if self.verbose: print("%s=%s" % (prob_query, normalised_dist))

        return normalised_dist

    # prints model performance according to the following metrics:
    # balanced accuracy, F1 score, AUC, Brier score, KL divergence,
    # and training and test times. But note that training time is
    # dependent on model training externally to this program, which
    # is the case of Bayes nets trained via CPT_Generator.py    
    def compute_performance(self, Y_true, Y_pred, Y_prob):
        #Constant To Avoid NAN In KL Divergence
        P = np.asarray(Y_true)+0.00001
        #Constant To Avoid NAN In KL Divergence
        Q = np.asarray(Y_prob)+0.00001
        
        bal_acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
        f1 = metrics.f1_score(Y_true, Y_pred)
        fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        brier = metrics.brier_score_loss(Y_true, Y_prob)
        kl_div = np.sum(P*np.log(P/Q))

        print("\nCOMPUTING performance on test data...")

        print("Balanced Accuracy="+str(bal_acc))
        print("F1 Score="+str(f1))
        print("Area Under Curve="+str(auc))
        print("Brier Score="+str(brier))
        print("KL Divergence="+str(kl_div))        
        print("Training Time=this number should come from the CPT_Generator!")
        print("Inference Time="+str(self.inference_time)+" secs.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: ModelEvaluator.py [config_file.txt] [test_file.csv]")
        print("EXAMPLE> ModelEvaluator.py config-lungcancer.txt lung_cancer-test.csv")
        exit(0)
    else:
        configfile = sys.argv[1]
        datafile_test = sys.argv[2]
        ModelEvaluator(configfile, datafile_test)