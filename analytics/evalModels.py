
import os, sys

from sklearn import model_selection
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
from sklearn.metrics import accuracy_score
sys.path.append(os.path.abspath("../"))

from analytics.Analyze import Analysis, SEED


def get_score_matrics(model:object, x_train:pd.DataFrame, y_train:pd.DataFrame, x_test:pd.DataFrame, y_test:pd.DataFrame) -> tuple:
    """Get the scoring matrics based on the input trained model

    Args:
        model (object): Trained input model
        x_train (pd.DataFrame): X train data split
        y_train (pd.DataFrame): Y training results
        x_test (pd.DataFrame): X test data split
        y_test (pd.DataFrame): Y testing results
    
    Returns:
        train_score, test_score: Scores determining the accuracy of the model
    """
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    return(round_value(metrics.f1_score(y_train,y_train_pred)),round_value(metrics.f1_score(y_test,y_test_pred)))


def splitAndScale(X: pd.DataFrame, Y: pd.DatetimeIndex,  test_size:float=0.3, scaleFlag:bool=False):
    """ Split the data and perform scalling if specified

    Args:
        X (DataFrame): Input features dataframe
        Y (DataFrame): Input results dataframe
    """
    # Split input data
    x_train_pre,x_test_pre,y_train_pre,y_test_pre = train_test_split(X,Y,test_size=test_size,random_state=42)

    if scaleFlag:
        # Perform Scalling
        scaler = StandardScaler().fit(x_train_pre)
        x_trainArray=scaler.transform(x_train_pre)
        x_train = pd.DataFrame(x_trainArray)
        x_train.columns= x_train_pre.columns

        x_testArray=scaler.transform(x_test_pre)
        x_test = pd.DataFrame(x_testArray)
        x_test.columns = x_test_pre.columns

        scaler = StandardScaler().fit(y_train_pre)
        y_trainArray=scaler.transform(y_train_pre)
        y_train = pd.DataFrame(y_trainArray)
        y_train.columns = y_train_pre.columns

        y_testArray=scaler.transform(y_test_pre)
        y_test = pd.DataFrame(y_testArray)
        y_test.columns =y_test_pre.columns

    else:
        x_train = x_train_pre
        x_test = x_test_pre
        y_train = y_train_pre
        y_test = y_test_pre
    
    return(x_train,y_train,x_test,y_test)  
    

def getBestfromGridSearch(classifier, x, y, param, scoringTechnique:str='f1'):
    clf = GridSearchCV(classifier(), param, cv=10,refit=True,scoring=scoringTechnique)
    clf.fit(x,y)
    return clf.best_estimator_, clf.best_params_


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

    fig = plt.figure()
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
    return


def generateLearningCurve(classifier,x,y,scoringTechnique="f1",outPath="./",base=False):

    train_sizes, train_scores, validation_scores = learning_curve(classifier,x,y,scoring=scoringTechnique,)
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)
    # print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
    print('\n', '-' * 20) # separator
    # print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

    fig = plt.figure()
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


def doDecisionTree(x_train,y_train,x_test,y_test,score,outPath) -> DecisionTreeClassifier:
    tree_para = {
        'max_depth':[2, 8, 16, 32, 48]
    }

    generateValidationCurve(DecisionTreeClassifier,
                        x_train,y_train,'max_depth',tree_para['max_depth'],scoringTechnique=score,outPath =outPath)

    validationInsights =  {
        'max_depth' :tree_para['max_depth'],
    }

    clf, best_params = getBestfromGridSearch(DecisionTreeClassifier,x_train,y_train,validationInsights,scoringTechnique=score)
    generateLearningCurve(clf,x_train,y_train,scoringTechnique=score,outPath =outPath)
    
    print("Feature Importances:")
    feature_importances = clf.feature_importances_
    sorted_indices =np.array(feature_importances).argsort()[::-1]
    print("Feature importance weights:",feature_importances)
    print("Columns Names Weight Sorted:", list(x_train.columns[sorted_indices]))
    print("\nBest Parameters:", best_params)

    y_pred = clf.predict(x_test)
    # print("AccuracyOptimized:",metrics.f1_score(y_test,y_pred))

    DecisionClassifier = DecisionTreeClassifier()
    DecisionClassifier.fit(x_train,y_train)
    generateLearningCurve(DecisionClassifier,x_train,y_train,scoringTechnique=score,outPath =outPath,base=True)

    # y_pred = DecisionClassifier.predict(x_test)
    # print("Accuracy2:",metrics.f1_score(y_test,y_pred))

    y_test_pred = clf.predict(x_test)
    y_train_pred = clf.predict(x_train)
    return(metrics.f1_score(y_train,y_train_pred),metrics.f1_score(y_test,y_test_pred))


def doAdaBoost(x_train,y_train,x_test,y_test,score,outPath,Dtree= None):

    # max_depths=[1,2,3,4,5,6,7,8,9,10,11,12,15,20]
    max_depths = [2, 8, 16, 32]
    dtClassifiers2=[]
    for depth in max_depths:
        dtClassifiers2.append(DecisionTreeClassifier(max_depth=depth))
    validationInsights =  {
        "n_estimators" : [4, 16, 48, 256],
        'base_estimator' : dtClassifiers2
    }
    print("Validating the estimators")
    generateValidationCurve(AdaBoostClassifier,
                            x_train,y_train,'n_estimators',validationInsights["n_estimators"],classifierParams=Dtree,scoringTechnique=score,outPath =outPath)
    print("Validating the base estimators")
    generateValidationCurve(AdaBoostClassifier,
                            x_train,y_train,'base_estimator',dtClassifiers2,plotRange=max_depths,scoringTechnique=score,outPath =outPath)

    # validationInsights =  {
    #     'n_estimators' :[10,20,30,40,50,60,70,80,90,100],
    #     'base_estimator' : dtClassifiers2
    # }

    print("Apply Grid Search")
    clf, best_params = getBestfromGridSearch(AdaBoostClassifier,x_train,y_train,validationInsights,scoringTechnique=score)
    generateLearningCurve(clf,x_train,y_train,scoringTechnique=score,outPath =outPath)


    print("Feature Importances:")
    feature_importances = clf.feature_importances_
    sorted_indices =np.array(feature_importances).argsort()[::-1]
    print("Feature importance weights:",feature_importances)
    print("Columns Names Weight Sorted:", list(x_train.columns[sorted_indices]))
    print("\nBest Parameters:", best_params)

    y_pred = clf.predict(x_test)

    print("AccuracyOptimized:",metrics.f1_score(y_test,y_pred))
    BoostClassifer = AdaBoostClassifier(n_estimators=50)
    BoostClassifer.fit(x_train,y_train)
    generateLearningCurve(BoostClassifer,x_train,y_train,scoringTechnique=score,outPath =outPath,base=True)
    y_pred = BoostClassifer.predict(x_test)
    print("Accuracy2:",metrics.f1_score(y_test,y_pred))
    y_train_pred = BoostClassifer.predict(x_train)

    y_test_pred = clf.predict(x_test)
    y_train_pred = clf.predict(x_train)
    return(metrics.f1_score(y_train,y_train_pred),metrics.f1_score(y_test,y_test_pred))


def doKNN(x_train,y_train,x_test,y_test,score,outPath):
    params = {
        "n_neighbors": [i for i in range(1,50)],
        # "weights": ["distance"]
        

    }
    generateValidationCurve(KNeighborsClassifier,x_train,y_train,'n_neighbors',params["n_neighbors"],scoringTechnique=score,outPath =outPath)

    clf, best_params = getBestfromGridSearch(KNeighborsClassifier,x_train,y_train,params,scoringTechnique=score)
    # clf.fit(x_train,y_train)

    print("\nBest Parameters:", best_params)


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
    hiddenLayers = [(5),(5,5),(10),(10,10),(10,10,10),(100),(100,5),(100,100),(100,100,100)]
    parameter_space = {
    'hidden_layer_sizes': hiddenLayers,
    'activation': ['tanh', 'relu','logistic'],
    'alpha': [0.0001, 0.001,0.05],
    'learning_rate': ['constant','adaptive'],
    'max_iter': [1000]
    }
    generateValidationCurve(MLPClassifier,x_train,y_train,'hidden_layer_sizes',parameter_space["hidden_layer_sizes"],[a for a in range(len(hiddenLayers))],scoringTechnique=score,outPath=outPath)
    generateValidationCurve(MLPClassifier,x_train,y_train,'activation',parameter_space["activation"],plotRange=parameter_space["activation"],scoringTechnique=score,outPath=outPath)
    generateValidationCurve(MLPClassifier,x_train,y_train,'alpha',parameter_space["alpha"],plotRange=[00.001,0.001,0.05],scoringTechnique=score,outPath=outPath)
    generateValidationCurve(MLPClassifier,x_train,y_train,'learning_rate',parameter_space["learning_rate"],plotRange=[0,1],scoringTechnique=score,outPath=outPath)
    generateValidationCurve(MLPClassifier,x_train,y_train,'max_iter',parameter_space["max_iter"],scoringTechnique=score)
    hiddenLayers = [(3),(3,3),(5,5),(5,5,5),(10),(10,10),(10,10,10)]
    parameter_space = {
    'hidden_layer_sizes': hiddenLayers,
    'activation': ['relu'],
    # 'solver': ['sgd', 'adam'],
    # 'alpha': [0.0001, 0.001,0.05],
    # 'learning_rate': ['constant','adaptive'],
    # 'max_iter':[i for i in range(200,2500,100)]
    'max_iter': [1000]
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
    plt.savefig(outPath+"/Iterative Learning Curve_"+MLPClassifier().__class__.__name__+"epoch")
    plt.clf()

    clf, best_params = getBestfromGridSearch(MLPClassifier,x_train,y_train,parameter_space,scoringTechnique=score)
    generateLearningCurve(clf,x_train,y_train,scoringTechnique=score,outPath =outPath)
    y_pred = clf.predict(x_test)

    print("\nBest Parameters:", best_params)

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


def doGradientBoosting(x_train,y_train,x_test,y_test,score,outPath,Dtree= None) -> GradientBoostingClassifier:

    validationInsights =  {
        "n_estimators" : [4, 16, 256, 512],
    }

    generateValidationCurve(GradientBoostingClassifier,x_train,y_train,'n_estimators',validationInsights["n_estimators"],classifierParams=Dtree,scoringTechnique=score,outPath =outPath)

    clf, best_params = getBestfromGridSearch(GradientBoostingClassifier,x_train,y_train,validationInsights,scoringTechnique=score)
    generateLearningCurve(clf,x_train,y_train,scoringTechnique=score,outPath =outPath)


    print("Feature Importances:")
    feature_importances = clf.feature_importances_
    sorted_indices =np.array(feature_importances).argsort()[::-1]
    print("Feature importance weights:",np.array(feature_importances))
    print("Columns Names Weight Sorted:", list(x_train.columns[sorted_indices]))
    print("\nBest Parameters:", best_params)

    y_pred = clf.predict(x_test)
    print("AccuracyOptimized:",metrics.f1_score(y_test,y_pred))

    BoostClassifer = GradientBoostingClassifier(n_estimators=50)
    BoostClassifer.fit(x_train,y_train)
    generateLearningCurve(BoostClassifer,x_train,y_train,scoringTechnique=score,outPath =outPath,base=True)

    y_pred = BoostClassifer.predict(x_test)
    print("Accuracy2:",metrics.f1_score(y_test,y_pred))
    y_train_pred = BoostClassifer.predict(x_train)

    y_test_pred = clf.predict(x_test)
    y_train_pred = clf.predict(x_train)

    # test_accuracy_train = accuracy_score(y_true=y_train, y_pred=y_train_pred)
    # test_accuracy_test = accuracy_score(y_true=y_test, y_pred=y_test_pred)
    return(metrics.f1_score(y_train,y_train_pred),metrics.f1_score(y_test,y_test_pred))
    # return(test_accuracy_train, test_accuracy_test)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    # plt.setp(bp['medians'], color=color, width=5)
    return


def round_value(value, digits:int=6):
    return round(value, digits)

if __name__ == "__main__":
    analyzeC = Analysis()
    df = analyzeC.create_data()

    CROSS_VALIDATION = False

    EVAL_MODELS = True

    EVAL_DECISION_TREE = True
    EVAL_ADA_BOOST = True
    EVAL_NN = True
    EVAL_KNN = True
    EVAL_GRADIENT_BOOSTING = True

    #### Evaluate based on half the data for faster computing
    Y = df["result"]
    X = df.drop(columns=["result"])
    # X, Y, _, _ = splitAndScale(X=X ,Y=Y ,test_size=0.5)

    columns1 = ['surface', 'best_of', 'indoor', 'elo_rating', 'rank', 'age', 'height', 'surface_win_p', 'indoor_p', 'best_of_win_p', 'matches_won_p', 'backhand']
    columns2 = ['surface', 'best_of', 'indoor', 'elo_rating', 'rank', 'surface_win_p', 'indoor_p', 'best_of_win_p', 'matches_won_p']
    columns3 = ['surface', 'best_of', 'indoor', 'elo_rating', 'rank', 'age', 'height']

    results = pd.DataFrame(columns=["FeatSet1","FeatSet2", "FeatSet3"])
    models = {
        "DT" : DecisionTreeClassifier(max_depth=2, random_state=SEED),
        "GradBoost" : GradientBoostingClassifier(n_estimators=256,random_state=SEED),
        "NN": MLPClassifier(activation="relu", max_iter=1000, solver="adam",warm_start=True,shuffle=True, random_state=SEED, alpha= 0.001, learning_rate='constant'),
        "AdaBoost": AdaBoostClassifier(n_estimators=256, random_state=SEED),
        "KNN" : KNeighborsClassifier(n_neighbors=6)
        }
    colors = {
        "FeatSet1" : "lightblue",
        "FeatSet2" : "pink", 
        "FeatSet3" : "lightgreen"
    }
    kfold = model_selection.KFold(n_splits=10, random_state=SEED, shuffle=True)
    ticks = list(models.keys())
    count = 1
    handles = []

    if CROSS_VALIDATION:

        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)

        for cols in [columns1, columns2, columns3]:
            label = "FeatSet" + str(count)
            print("\n--> Working on cols:", len(cols), cols)

            x_train,y_train,x_test,y_test = splitAndScale(X[cols],Y) 
            print(len(x_train.columns.tolist()))

            resultDict = dict()
            for modelName, model in models.items():
                print("Model:", modelName)
                cv_results = model_selection.cross_val_score(model, X[cols], Y, cv=kfold, scoring="accuracy")
                resultDict[modelName] = cv_results
            results[label] = resultDict

            data = list(resultDict.values())
            print(label, data)
            
            featSetBplt = plt.boxplot(data,  positions=np.array(range(len(data)))*3.0+count*0.6, patch_artist=True, labels=ticks)
            set_box_color(featSetBplt, colors[label])
            handles.append(featSetBplt)
            count += 1
            # del x_train
            # del y_train
            # del x_test
            # del y_test

        # results.boxplot()
        
        # ax.set_xticklabels(list(models.keys()))
        # ax.set_xticks()
        # plt.show()

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.xticks(range(1, len(ticks) * 3, 3), ticks)
        plt.legend(list(colors.keys()), loc="center right", bbox_to_anchor=(1.4, 0.5),   fancybox=True)
        leg = ax.get_legend()
        hl_dict = {handle.get_label(): handle for handle in leg.legendHandles}
        for handleIndex, handle in enumerate(leg.legendHandles):
            handle.set_color(colors["FeatSet"+str(handleIndex+1)])

        plt.grid(True)
        plt.savefig("./Images/EvalModels_CV_FeatSets") 

    else:
        x_train,y_train,x_test,y_test = splitAndScale(X,Y) 

    if EVAL_MODELS:
        #### Evaluating the models best parameters
        fp = open("experiments.csv", "w")
        fp.write("ML Model, Train Accuracy, Test Accuracy, Best Optimization Parameters\n")

        if EVAL_DECISION_TREE:
            print("\nRunning Decision Tree:")

            validationInsights =  {
                'max_depth':[2, 8, 16, 32, 48],
            }
            clf, best_params = getBestfromGridSearch(DecisionTreeClassifier,x_train,y_train,validationInsights,scoringTechnique="f1")
            best_params = str(best_params)
            train_score, test_score = get_score_matrics(clf, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            fp.write("Decision Tree,"+str(train_score)+","+str(test_score)+","+best_params+"\n")

            train_score, test_score = doDecisionTree(x_train,y_train,x_test,y_test,"f1","./Images/")
            # train_score = round_value(train_score)
            # test_score = round_value(test_score)
            # print("Train Score:", train_score)
            # print("Test Score:", test_score)

        if EVAL_GRADIENT_BOOSTING:
            print("\nRunning Gradient Boosting")
            validationInsights =  {
                "n_estimators" : [4, 16, 48, 256, 512],
            }
            clf, best_params = getBestfromGridSearch(GradientBoostingClassifier,x_train,y_train,validationInsights,scoringTechnique="f1")
            best_params = str(best_params)
            train_score, test_score = get_score_matrics(clf, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            fp.write("Gradient Boosting,"+str(train_score)+","+str(test_score)+","+best_params+"\n")
            
            train_score, test_score = doGradientBoosting(x_train,y_train,x_test,y_test,"f1","./Images/")
            # train_score, test_score = round_value(train_score), round_value(test_score)
            # print("Train Score:", train_score)
            # print("Test Score:", test_score)

        if EVAL_NN:
            print("\nRunning NN")
            # hiddenLayers = [(5),(5,5),(10),(10,10),(10,10,10),(100),(100,5),(100,100),(100,100,100)]
            hiddenLayers = [(5,5),(10,10),(100,100,100), (1000,100,1000)]
            validationInsights = {
                # 'hidden_layer_sizes': hiddenLayers,
                'activation': ['tanh', 'relu','logistic'],
                'alpha': [0.001,0.05,0.1],
                'learning_rate': ['constant','adaptive'],
                'max_iter': [1000]
            }
            clf, best_params = getBestfromGridSearch(MLPClassifier,x_train,y_train,validationInsights,scoringTechnique="f1")
            best_params = str(best_params)
            train_score, test_score = get_score_matrics(clf, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            fp.write("NN,"+str(train_score)+","+str(test_score)+","+best_params+"\n")

            train_score, test_score = doNN(x_train,y_train,x_test,y_test,"f1","./Images/")
            # train_score = round(train_score, 6)
            # test_score = round(test_score, 6)
            # print("Train Score:", train_score)
            # print("Test Score:", test_score)

        if EVAL_ADA_BOOST:
            print("\nRunning Ada Boost:")
            max_depths = [2, 8, 16, 32]
            dtClassifiers2=[]
            for depth in max_depths: dtClassifiers2.append(DecisionTreeClassifier(max_depth=depth))
            validationInsights =  {
                "n_estimators" : [4, 16, 256],
                'base_estimator' : dtClassifiers2
            }
            clf, best_params = getBestfromGridSearch(AdaBoostClassifier,x_train,y_train,validationInsights,scoringTechnique="f1")
            best_params = str(best_params)
            train_score, test_score = get_score_matrics(clf, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            fp.write("Ada Boost,"+str(train_score)+","+str(test_score)+","+best_params+"\n")

            train_score, test_score = doAdaBoost(x_train,y_train,x_test,y_test,"f1","./Images/")
            # train_score = round(train_score, 6)
            # test_score = round(test_score, 6)
            # print("Train Score:", train_score)
            # print("Test Score:", test_score)

        if EVAL_KNN:
            print("\nRunning kNN:")
            validationInsights = {
                "n_neighbors": [i for i in range(1,100, 5)],
            }
            clf, best_params = getBestfromGridSearch(KNeighborsClassifier,x_train,y_train,validationInsights,scoringTechnique="f1")
            best_params = str(best_params)
            train_score, test_score = get_score_matrics(clf, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            fp.write("KNN,"+str(train_score)+","+str(test_score)+","+best_params+"\n")

            train_score, test_score = doKNN(x_train,y_train,x_test,y_test,"f1","./Images/")
            # train_score = round(train_score, 6)
            # test_score = round(test_score, 6)
            # print("Train Score:", train_score)
            # print("Test Score:", test_score)

        fp.close()
