import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
svd = TruncatedSVD(n_components=5, random_state=42)
svd.fit(X) 
TruncatedSVD(algorithm='randomized', n_components=5, n_iter=5,
        random_state=42, tol=0.0)
print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum()) 
