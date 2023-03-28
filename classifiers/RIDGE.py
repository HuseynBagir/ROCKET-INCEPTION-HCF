import numpy as np

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score

class RIDGE:

    def __init__(self, alphas = np.logspace(-3, 3, 10), normalize = True):

        self.alphas = alphas
        self.normalize = normalize

        self.clf = RidgeClassifierCV(alphas=self.alphas)#, normalize=self.normalize)
    
    def fit(self, X, Y):

        self.clf.fit(X, Y)
    
    def predict(self, X, Y):

        ypred = self.clf.predict(X)
        
        return accuracy_score(y_true=Y, y_pred=ypred)#, normalize=True)
    
    
    