from pystruct.datasets import load_letters
import numpy as np
letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
X, y = np.array(X), np.array(y)
X_train, X_test = X[folds == 1], X[folds != 1]
y_train, y_test = y[folds == 1], y[folds != 1]


features, y, folds = letters['data'], letters['labels'], letters['folds']
features, y = np.array(features), np.array(y)
features_train, features_test = features[folds == 1], features[folds != 1]
y_train, y_test = y[folds == 1], y[folds != 1]

"""
features_0 = features_train[0]
n_nodes = features_0.shape[0]
edges_0 = np.vstack([np.arange(n_nodes - 1), np.arange(1, n_nodes)])
x = (features_0, edges_0)
"""

f_t = features_train
X_train = [(features_i, np.vstack([np.arange(features_i.shape[0] - 1), np.arange(1, features_i.shape[0])])) for features_i in f_t]



print type(X_train)
print type(X_train[0][1])
print X_train[0][1].shape

print type(y_train)
print type(y_train[0])
print y_train[0]
print y_train[0].shape

from pystruct.models import GraphCRF
from pystruct.learners import FrankWolfeSSVM
model = GraphCRF(directed=True, inference_method="max-product")
ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)
ssvm.fit(X_train, y_train)
print "OM SRI SAIRAM"
print ("Accuracy score with Graph CRF : %f" % ssvm.score(y_train,y_test))
