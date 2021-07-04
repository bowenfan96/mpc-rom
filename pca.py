import numpy as np
from sklearn.decomposition import PCA

A = np.genfromtxt('A.csv', delimiter=',')
pca = PCA()
pca.fit_transform(A)

np.set_printoptions(suppress=True)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

assert np.sum(pca.explained_variance_ratio_) == 1