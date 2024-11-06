#############################################################################
# BayesNetReader.py
#
# Reads a configuration file containing the specification of a Bayes net.
# It generates a dictionary of key-value pairs containing information
# describing the random variables, structure, and conditional probabilities.
# This implementation aims to be agnostic of the data (no hardcoded vars/probs)
## 
# Keys expected: name, random_variables, structure, and CPTs.
# Separators: COLON(:) for key-values, EQUALS(=) for table_entry-probabilities
# The following is a snippet of configuration file config_alarm.txt
# --------------------------------------------------------
# name:Alarm
# 
# random_variables:Burglary(B);Earthquake(E);Alarm(A);JohnCalls(J);MaryCalls(M)
# 
# structure:P(B);P(E);P(A|B,E);P(J|A);P(M|A)
# 
# CPT(B):
# true=0.001;false=0.999
#
# CPT(E):
# true=0.002;false=0.998
# 
#  ...
# 
# CPT(M|A):
# true|true=0.70;
# true|false=0.01;
# false|true=0.30;
# false|false=0.99
# --------------------------------------------------------
# The file above replaces CPTs by regression_models in the case of Bayes nets
# with continuous data, where instead of CPTs regression models are used.
#
# Version: 1.0
# Date: 06 October 2022 first version
# Date: 25 October 2023 extended for Bayes nets with KernelRidge regression
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import pickle


class BayesNetReader:
    def __init__(self, file_name):
        self.bn = {}  # Make bn an instance variable
        self.read_data(file_name)
        self.tokenise_data()

    # starts loading a configuration file into dictionary 'bn', by
    # splitting strings with character ':' and storing keys and values 
    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))

        try:
            with open(data_file, encoding='utf-8-sig') as cfg_file:
                key = None
                value = None
                for line in cfg_file:
                    line = line.strip().replace('\ufeff', '')
                    if len(line) == 0:
                        continue

                    tokens = line.split(":")
                    if len(tokens) == 2:
                        if value is not None:
                            self.bn[key] = value
                            value = None

                        key = tokens[0].replace('\ufeff', '')
                        value = tokens[1].replace('\ufeff', '')
                    else:
                        value += tokens[0].replace('\ufeff', '')

                if key and value is not None:  # Ensure the last key-value pair is added
                    self.bn[key] = value
                self.bn["random_variables_raw"] = self.bn.get("random_variables", "")
                print("RAW key-values=" + str(self.bn))
        except FileNotFoundError:
            print(f"Error: The file '{data_file}' was not found.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)

    # continues loading a configuration file into dictionary 'bn', by
    # separating key-value pairs as follows:
    # (a) random_variables are stored as list in self.bn['random_variables']
    # (b) CPTs are stored as an inner dictionary in self.bn['CPT']
    # (c) all others are stored as key-value pairs in self.bn[key]
    def tokenise_data(self):
        print("TOKENISING data...")
        rv_key_values = {}

        for key, values in self.bn.items():
            if key == "random_variables":
                var_set = []
                for value in values.split(";"):
                    if value.find("(") > -1 and value.find(")") > -1:
                        value = value.replace('(', ' ').replace(')', ' ').replace('\ufeff', '')
                        parts = value.split(' ')
                        var_set.append(parts[1].strip())
                    else:
                        var_set.append(value)
                self.bn[key] = var_set

            elif key.startswith("CPT"):
                # store Conditional Probability Tables (CPTs) as dictionaries
                cpt = {}
                total_prob = 0  # Renamed from sum to avoid shadowing built-in function
                for value in values.split(";"):
                    pair = value.split("=")
                    if len(pair) == 2:  # Check for valid key-value pairs
                        cpt[pair[0].replace('\ufeff', '')] = float(pair[1].replace('\ufeff', ''))
                        total_prob += float(pair[1].replace('\ufeff', ''))
                print("key=%s cpt=%s total_prob=%s" % (key, cpt, total_prob))
                self.bn[key] = cpt

                # store unique values for each random variable
                rand_var = key[4:].split("|")[0] if "|" in key else key[4:].split(")")[0]
                unique_values = list(cpt.keys())
                rv_key_values[rand_var.replace('\ufeff', '')] = unique_values

            else:
                values = [val.replace('\ufeff', '') for val in values.split(";")]
                if len(values) > 1:
                    self.bn[key] = values

        self.bn['rv_key_values'] = rv_key_values
        print("TOKENISED key-values=" + str(self.bn))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: BayesNetReader.py [your_config_file.txt]")
    else:
        file_name = sys.argv[1]
        BayesNetReader(file_name)