import pandas as pd
from sklearn import decomposition

data = pd.read_csv("data/pca_training_data.csv").to_numpy()
pca = decomposition.PCA(n_components=20)
pca.fit(data)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
