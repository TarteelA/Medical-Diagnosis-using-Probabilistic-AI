#Import Libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

X_train = np.array([
    [0.696, 0.883],[0.681, 0.876],
    [0.736, 1.046],[0.713, 1.01],
    [0.701, 1.034],[0.71, 1.444],
    [0.718, 1.462],[0.712, 1.039],
    [0.711, 1.032],[0.705, 1.033]
])

X_test = np.array([[0.769, 1.213],[0.697, 1.096]])
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

plt.figure(figsize=(8, 6))
sns.heatmap(kernel_vectors, annot=True, cmap='viridis', square=True, cbar=True)
plt.title("RBF Kernel Vectors")
plt.xlabel("X_train")
plt.ylabel("X_test")
plt.show()