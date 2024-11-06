#Import Libraries
import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

#Choices Of Scoring Functions: bic, k2, bdeu, bds, aic
TRAINING_DATA = './data/dementia_data-MRI-features.csv'
SCORING_FUNCTION = 'bic'
MAX_ITERATIONS=20000000
VISUALISE_STRUCTURE=True

#Data Loading Using Pandas
data = pd.read_csv(TRAINING_DATA, encoding='UTF-8', nrows=374)
data = data.dropna()
data = data[~data.isin([float('inf'), -float('inf')]).any(axis=1)]
print("DATA:\n", data)

# definition of directed acyclic graph (predefined Naive Bayes structure -- only for discretising data)
edges = [('CDR', 'Subject_ID'),('CDR', 'MRI_ID'),('CDR', 'Group'),('CDR', 'Visit'),
         ('CDR', 'MR_Delay'),('CDR', 'M/F'),('CDR', 'Hand'),('CDR', 'Age'),
		 ('CDR', 'EDUC'),('CDR', 'SES'),('CDR', 'MMSE'),('CDR', 'eTIV'),('CDR', 'nWBV'),('CDR', 'ASF')]

#Performs Discretisation Of Continuous Data For Columns Specified and Structure Provided Used For Training Bayesian Network
continuous_columns = ["MR_Delay", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
discrete_data = bn.discretize(data, edges, continuous_columns, max_iterations=1, verbose=3)
for randvar in discrete_data:
    print("VARIABLE:",randvar)
    print(discrete_data[randvar])

#Structure Learning Using Chosen Scoring Function as per SCORING_FUNCTION
model = bn.structure_learning.fit(discrete_data, methodtype='hillclimbsearch', scoretype=SCORING_FUNCTION, max_iter=MAX_ITERATIONS)
num_model_edges = len(model['model_edges'])
print("model=",model)
print("num_model_edges="+str(num_model_edges))

#Visualise The Learnt Structure
if VISUALISE_STRUCTURE:
    G = nx.DiGraph()
    G.add_edges_from(model['model_edges'])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightgreen', font_size=10, arrows=True)
    plt.title('Directed Acyclic Graph (DAG)')
    plt.show()

#Parameter Learning Using Maximum Likelihood Estimation (MLE) and Discretised Data
DAG = bn.make_DAG(model['model_edges'])
model = bn.parameter_learning.fit(DAG, discrete_data, methodtype="maximumlikelihood")
print("model=",model)

#Probabilistic Inference For Test Example
discretised_evidence = { 
'MR_Delay':bn.discretize_value(discrete_data["MR_Delay"], 457), 
'Age':bn.discretize_value(discrete_data["Age"], 88), 
'EDUC':bn.discretize_value(discrete_data["EDUC"], 14), 
'SES':bn.discretize_value(discrete_data["SES"], 2), 
'MMSE':bn.discretize_value(discrete_data["MMSE"], 30),
'CDR':bn.discretize_value(discrete_data["CDR"], 0),
'eTIV':bn.discretize_value(discrete_data["eTIV"], 2004),
'nWBV':bn.discretize_value(discrete_data["nWBV"], 0.681),
'ASF':bn.discretize_value(discrete_data["ASF"], 0.876),
'Subject_ID':bn.discretize_value(discrete_data["Subject_ID"], 'OAS2_0001'), 
'MRI_ID':bn.discretize_value(discrete_data["MRI_ID"], 'OAS2_0001_MR2'),
'Group':bn.discretize_value(discrete_data["Group"], 'Nondemented'),
'Visit':bn.discretize_value(discrete_data["Visit"], 2),
'M/F':bn.discretize_value(discrete_data["M/F"], 'M'),
'Hand':bn.discretize_value(discrete_data["Hand"], 'R')}

inference_result = bn.inference.fit(model, variables=['CDR'], evidence=discretised_evidence)
print(inference_result)