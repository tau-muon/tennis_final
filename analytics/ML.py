import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import sys, os

sys.path.append(os.path.abspath("../"))
from analytics.Analyze import Analysis


class MLModel(object):
    def __init__(self) -> None:
        self.a = Analysis()
    
    def use_decisiontree(self):
        self.model = DecisionTreeClassifier()

    # def use_randomforest(self):
    #     self.model = 

    def train(self, X, y):
        self.model.fit(X, y)
        return

    def predict_proba(self, X):
        return self.model.predict_proba(X=X)
    
    
if __name__ == "__main__":
    a = Analysis()
    dt = DecisionTreeClassifier()
    dt.fit(a.X, a.y)

    # Predict
    x_predict = []
    print(dt.predict_proba(x_predict))

