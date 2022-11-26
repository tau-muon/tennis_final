
import os, sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.exceptions import ConvergenceWarning
import time
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
sys.path.append(os.path.abspath("../"))

from analytics.Analyze import Analysis
def splitAndScale(X,Y):
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

    # scaler = StandardScaler().fit(x_trainPre)
    # x_trainArray=scaler.transform(x_trainPre)
    # x_train = pd.DataFrame(x_trainArray)
    # x_train.columns=x_trainPre.columns

    # x_testArray=scaler.transform(x_testPre)
    # x_test = pd.DataFrame(x_testArray)
    # x_test.columns=x_testPre.columns  
    return(x_train,y_train,x_test,y_test)  
    

def getBestfromGridSearch(classifier,x,y,param,scoringTechnique='f1'):
    clf = GridSearchCV(classifier(), param, cv=5,refit=True,scoring=scoringTechnique)
    clf.fit(x,y)
    print("Best Parameters from GridSearch for:"+classifier().__class__.__name__+"\n")
    print(clf.best_params_)
    return clf.best_estimator_


def generateValidationCurve(classifer,x,y,paramName,paramRange,plotRange=None,scoringTechnique="f1",outPath="./",classifierParams=None):
    if plotRange == None:
        plotRange=paramRange
    train_scores, test_scores = validation_curve(
        classifer(),
        x,
        y,
        param_name=paramName,
        param_range=paramRange,
        scoring=scoringTechnique,
        n_jobs=2,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve for " + classifer().__class__.__name__)
    plt.xlabel(paramName)
    plt.ylabel("Score")
    # plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(
        plotRange, train_scores_mean, label="Training score", color="darkorange", lw=lw
    )
    plt.fill_between(
        plotRange,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.plot(
        plotRange, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
    )
    plt.fill_between(
        plotRange,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.savefig(outPath+"./Validation Curve"+"_"+classifer().__class__.__name__+"_"+paramName)
    # plt.show()    
    plt.clf()

def generateLearningCurve(classifier,x,y,scoringTechnique="f1",outPath="./",base=False):

    train_sizes, train_scores, validation_scores = learning_curve(classifier,x,y,scoring=scoringTechnique,)
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)
    # print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
    print('\n', '-' * 20) # separator
    # print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('Score', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves for '+ classifier.__class__.__name__, fontsize = 18, y = 1.03)
    plt.legend()
    # plt.ylim(0,40)
    if not base:
        plt.savefig(outPath+"./Learning Curve"+"_"+classifier.__class__.__name__)
    else:
        plt.savefig(outPath+"./Learning Curve"+"_"+classifier.__class__.__name__+"BaseImplementation")
    # plt.show()
    plt.clf()


def doDecisionTree(x_train,y_train,x_test,y_test,score,outPath):
    path = DecisionTreeClassifier()
    # ccp_alphas = path.ccp_alphas
    tree_para = {
        'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,15,20],
        # 'ccp_alpha':ccp_alphas,

        }

    generateValidationCurve(DecisionTreeClassifier,x_train,y_train,'max_depth',tree_para['max_depth'],scoringTechnique=score,outPath =outPath)


    # generateValidationCurve(DecisionTreeClassifier,x_train,y_train,'ccp_alpha',tree_para['ccp_alpha'],scoringTechnique=score,outPath =outPath)



    validationInsights =  {
        'max_depth' :tree_para['max_depth'],
        # 'ccp_alpha' : ccp_alphas
    }


    clf = getBestfromGridSearch(DecisionTreeClassifier,x_train,y_train,validationInsights,scoringTechnique=score)
    generateLearningCurve(clf,x_train,y_train,scoringTechnique=score,outPath =outPath)


    y_pred = clf.predict(x_test)

    print("AccuracyOptimized:",metrics.f1_score(y_test,y_pred))

    DecisionClassifier = DecisionTreeClassifier()
    DecisionClassifier.fit(x_train,y_train)
    generateLearningCurve(DecisionClassifier,x_train,y_train,scoringTechnique=score,outPath =outPath,base=True)

    y_pred = DecisionClassifier.predict(x_test)
    print("Accuracy2:",metrics.f1_score(y_test,y_pred))
    y_pred = clf.predict(x_test)

    y_train_pred = clf.predict(x_train)
    return(metrics.f1_score(y_train,y_train_pred),metrics.f1_score(y_test,y_pred))


if __name__ == "__main__":
    analyzeC = Analysis()
    df = analyzeC.create_data()
    print(df)
    
    Y = df['result']
    print(Y)
    X = df.drop(columns=['result'])
    x_train,y_train,x_test,y_test = splitAndScale(X,Y) 
    doDecisionTree(x_train,y_train,x_test,y_test,"f1",".")



