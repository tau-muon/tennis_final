
import os, sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
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
        'max_depth':[1,2,3,4,5,6,7,8,9,10, 50, 100],
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

    
    print("Feature Importances:")
    feature_importances = clf.feature_importances_
    sorted_indices =np.array(feature_importances).argsort()[::-1]
    print(feature_importances)
    print(sorted_indices)
    print(x_train.columns[sorted_indices])

    print("AccuracyOptimized:",metrics.f1_score(y_test,y_pred))

    DecisionClassifier = DecisionTreeClassifier()
    DecisionClassifier.fit(x_train,y_train)
    generateLearningCurve(DecisionClassifier,x_train,y_train,scoringTechnique=score,outPath =outPath,base=True)

    y_pred = DecisionClassifier.predict(x_test)
    # print("Accuracy2:",metrics.f1_score(y_test,y_pred))


    y_test_pred = clf.predict(x_test)
    y_train_pred = clf.predict(x_train)
    return(metrics.f1_score(y_train,y_train_pred),metrics.f1_score(y_test,y_test_pred))


def doAdaBoost(x_train,y_train,x_test,y_test,score,outPath,Dtree= None):

    max_depths=[1,2,3,4,5,6,7,8,9,10,11,12,15,20]
    dtClassifiers2=[]
    for depth in max_depths:
        dtClassifiers2.append(DecisionTreeClassifier(max_depth=depth))
    generateValidationCurve(AdaBoostClassifier,x_train,y_train,'n_estimators',[1,3,5,10,20,30,40,90,100,150,200,250,300],classifierParams=Dtree,scoringTechnique=score,outPath =outPath)
    generateValidationCurve(AdaBoostClassifier,x_train,y_train,'base_estimator',dtClassifiers2,plotRange=max_depths,scoringTechnique=score,outPath =outPath)


    updatedClassifiers = []

    updatedClassifiers.extend(dtClassifiers2)
    validationInsights =  {
        'n_estimators' :[10,20,30,40,50,60,70,80,90,100],
        'base_estimator' : dtClassifiers2
    }

    clf = getBestfromGridSearch(AdaBoostClassifier,x_train,y_train,validationInsights,scoringTechnique=score)
    generateLearningCurve(clf,x_train,y_train,scoringTechnique=score,outPath =outPath)


    y_pred = clf.predict(x_test)

    print("AccuracyOptimized:",metrics.f1_score(y_test,y_pred))

    BoostClassifer = AdaBoostClassifier(n_estimators=50)
    BoostClassifer.fit(x_train,y_train)
    generateLearningCurve(BoostClassifer,x_train,y_train,scoringTechnique=score,outPath =outPath,base=True)


    y_pred = BoostClassifer.predict(x_test)
    # print("Accuracy2:",metrics.f1_score(y_test,y_pred))
    y_train_pred = BoostClassifer.predict(x_train)
    return(metrics.f1_score(y_train,y_train_pred),metrics.f1_score(y_test,y_pred))

def doKNN(x_train,y_train,x_test,y_test,score,outPath):
    params = {
        "n_neighbors": [i for i in range(1,50)],
        # "weights": ["distance"]
        

    }
    generateValidationCurve(KNeighborsClassifier,x_train,y_train,'n_neighbors',params["n_neighbors"],scoringTechnique=score,outPath =outPath)

    clf = getBestfromGridSearch(KNeighborsClassifier,x_train,y_train,params,scoringTechnique=score)
    # clf.fit(x_train,y_train)
    generateLearningCurve(clf,x_train,y_train,scoringTechnique=score,outPath =outPath)
    
    y_pred = clf.predict(x_test)

    print("AccuracyOptimized:",metrics.f1_score(y_test,y_pred))
    knnClassifier = KNeighborsClassifier()

    knnClassifier.fit(x_train,y_train)
    generateLearningCurve(knnClassifier,x_train,y_train,scoringTechnique=score,outPath =outPath,base=True)

    y_pred = knnClassifier.predict(x_test)
    print("Accuracy2:",metrics.f1_score(y_test,y_pred))
    y_train_pred = clf.predict(x_train)
    y_pred = clf.predict(x_test)

    return(metrics.f1_score(y_train,y_train_pred),metrics.f1_score(y_test,y_pred))

def doNN(x_train,y_train,x_test,y_test,score,outPath):
    # hiddenLayers= [x for x in itertools.product(range(1,10), range(1,10),range(0,10))]
    hiddenLayers = [(5),(5,5),(5,5,5),(10),(10,10),(10,10,10),(100),(100,5),(100,100),(100,100,100)]
    parameter_space = {
    'hidden_layer_sizes': hiddenLayers,
    'activation': ['tanh', 'relu','logistic'],
    'alpha': [0.0001, 0.001,0.05],
    'learning_rate': ['constant','adaptive'],
    'max_iter': [220]
    }
    generateValidationCurve(MLPClassifier,x_train,y_train,'hidden_layer_sizes',parameter_space["hidden_layer_sizes"],[a for a in range(len(hiddenLayers))],scoringTechnique=score,outPath=outPath)
    generateValidationCurve(MLPClassifier,x_train,y_train,'activation',parameter_space["activation"],plotRange=parameter_space["activation"],scoringTechnique=score,outPath=outPath)
    # generateValidationCurve(MLPClassifier,x_train,y_train,'alpha',parameter_space["alpha"],plotRange=[00.001,0.001,0.05],scoringTechnique=score,outPath=outPath)
    # generateValidationCurve(MLPClassifier,x_train,y_train,'learning_rate',parameter_space["learning_rate"],plotRange=[0,1],scoringTechnique=score,outPath=outPath)
    # generateValidationCurve(MLPClassifier,x_train,y_train,'max_iter',parameter_space["max_iter"],scoringTechnique=score)
    hiddenLayers = [(3),(3,3),(5,5),(5,5,5),(10),(10,10),(10,10,10)]
    parameter_space = {
    'hidden_layer_sizes': hiddenLayers,
    'activation': ['relu'],
    # 'solver': ['sgd', 'adam'],
    # 'alpha': [0.0001, 0.001,0.05],
    # 'learning_rate': ['constant','adaptive'],
    # 'max_iter':[i for i in range(200,2500,100)]
    'max_iter': [i for i in range(80,140,10)]
    }
    mlp = MLPClassifier(activation="relu", max_iter=10, solver="adam",warm_start=True)

    epochs = 600
    training_mse = []
    validation_mse = []
    x_trainEpoch,xvalidate,y_trainEpoch,y_validate= train_test_split(x_train,y_train,test_size=0.35,random_state=42)
    for epoch in range(epochs):
        mlp.fit(x_trainEpoch, y_trainEpoch) 
        Y_pred = mlp.predict(x_trainEpoch)
        curr_train_score = mean_squared_error(y_trainEpoch, Y_pred) # training performances
        Y_pred = mlp.predict(xvalidate) 
        curr_valid_score = mean_squared_error(y_validate, Y_pred) # validation performances
        training_mse.append(curr_train_score) # list of training perf to plot
        validation_mse.append(curr_valid_score) # list of valid perf to plot
    
    plt.title("iterative Learning curve for " + MLPClassifier().__class__.__name__)
    plt.xlabel('Epoch')
    plt.ylabel("Mean Squared Error")
    plt.plot(training_mse,label="training mse", color="darkorange")
    plt.plot(validation_mse,label="validation mse", color="blue")
    plt.legend(loc="best")
    plt.savefig(outPath+"Iterative Learning Curve_"+MLPClassifier().__class__.__name__+"epoch")
    plt.clf()

    clf = getBestfromGridSearch(MLPClassifier,x_train,y_train,parameter_space,scoringTechnique=score)
    generateLearningCurve(clf,x_train,y_train,scoringTechnique=score,outPath =outPath)
    y_pred = clf.predict(x_test)

    print("Feature Importances:")
    print(clf.feature_importances_)
    print("AccuracyOptimized:",metrics.f1_score(y_test,y_pred))

    NNClassifier = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
    NNClassifier.fit(x_train,y_train)
    generateLearningCurve(NNClassifier,x_train,y_train,scoringTechnique=score,outPath =outPath,base=True)

    y_pred = NNClassifier.predict(x_test)
    # print("Accuracy:",metrics.f1_score(
    # y_test,y_pred))
    y_train_pred = clf.predict(x_train)
    y_pred = clf.predict(x_test)

    return(metrics.f1_score(y_train,y_train_pred),metrics.f1_score(y_test,y_pred))


def doGradientBoosting(x_train,y_train,x_test,y_test,score,outPath,Dtree= None):

    max_depths=[1,2,3,4,5,6,7,8,9,10,11,12,15,20]
    dtClassifiers2=[]
    for depth in max_depths:
        dtClassifiers2.append(DecisionTreeClassifier(max_depth=depth))
    generateValidationCurve(GradientBoostingClassifier,x_train,y_train,'n_estimators',[1,3,5,10,20,30,40,50,60,70,80,90,100,150,200,250,300],classifierParams=Dtree,scoringTechnique=score,outPath =outPath)
    generateValidationCurve(GradientBoostingClassifier,x_train,y_train,'base_estimator',dtClassifiers2,plotRange=max_depths,scoringTechnique=score,outPath =outPath)


    updatedClassifiers = []

    updatedClassifiers.extend(dtClassifiers2)
    validationInsights =  {
        'n_estimators' :[10,20,30,40,50,60,70,80,90,100],
        'base_estimator' : dtClassifiers2
    }

    clf = getBestfromGridSearch(GradientBoostingClassifier,x_train,y_train,validationInsights,scoringTechnique=score)
    generateLearningCurve(clf,x_train,y_train,scoringTechnique=score,outPath =outPath)


    y_pred = clf.predict(x_test)

    print("Feature Importances:")
    feature_importances = clf.feature_importances_
    sorted_indices =np.array(feature_importances).argsort()[::-1]
    print(feature_importances)
    print(sorted_indices)
    print(x_train.columns[sorted_indices])


    print("AccuracyOptimized:",metrics.f1_score(y_test,y_pred))

    BoostClassifer = GradientBoostingClassifier(n_estimators=50)
    BoostClassifer.fit(x_train,y_train)
    generateLearningCurve(BoostClassifer,x_train,y_train,scoringTechnique=score,outPath =outPath,base=True)


    y_pred = BoostClassifer.predict(x_test)
    print("Accuracy2:",metrics.f1_score(y_test,y_pred))
    y_train_pred = BoostClassifer.predict(x_train)
    return(metrics.f1_score(y_train,y_train_pred),metrics.f1_score(y_test,y_pred))


if __name__ == "__main__":
    analyzeC = Analysis()
    df = analyzeC.create_data()
    df.to_csv("./data_debug.csv")

    Y = df['result']
    X = df.drop(columns=['result'])
    x_train,y_train,x_test,y_test = splitAndScale(X,Y) 

    # train_score, test_score = doDecisionTree(x_train,y_train,x_test,y_test,"f1",".")

    # doGradientBoosting(x_train,y_train,x_test,y_test,"f1",".")
        
    train_score, test_score = doNN(x_train,y_train,x_test,y_test,"f1",".")

    # train_score, test_score = doAdaBoost(x_train,y_train,x_test,y_test,"f1",".")
    # doKNN(x_train,y_train,x_test,y_test,"f1",".")

    print("Train Score:", train_score)
    print("Test Score:", test_score)
    




