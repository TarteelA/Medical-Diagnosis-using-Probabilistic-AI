#Import Libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Declare X_train And X_test
X_train = np.array([
    [2.301442,0.284654],[2.486855,0.368674],
    [2.342259,0.332634],[2.405554,0.368975],
    [2.33218,0.410335],[2.18756,0.357775],
    [1.854785,0.211756],[2.064693,0.163755],
    [2.322511,0.231571],[2.432792,0.271362]
])

X_test = np.array([[3.007463,0.430788],[2.846369,0.219514]])

#Compute RBF Kernel Vector
gamma = 0.5

def rbf_kernel_vector(x_test, X_train, gamma):
    kernel_vector = np.array([
        np.exp(-gamma * np.sum((x_test - x_train) ** 2))
        for x_train in X_train
    ])
    return kernel_vector

kernel_vectors = []
for x_test in X_test:
    kernel_vector = rbf_kernel_vector(x_test, X_train, gamma)
    print(f"RBF kernel vector for {x_test}:", kernel_vector)
    kernel_vectors.append(kernel_vector)

diff = np.abs(kernel_vector[0]-kernel_vector[1])
print("gamma=%s diff=%s" % (gamma, diff))

#Show Heatmap Of RBF Kernel Vector
plt.figure(figsize=(8, 6))
sns.heatmap(kernel_vectors, annot=True, cmap='viridis', square=True, cbar=True)
plt.title("RBF Kernel Vectors")
plt.xlabel("X_train")
plt.ylabel("X_test")
plt.show()