import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import sys, os
import pickle

sys.path.append(os.path.abspath("../"))
from analytics.FeatEng import DB_PATH
from analytics.Analyze import Analysis


class MLModel(object):
    def __init__(self) -> None:
        analysisI = Analysis(mode="prod")
        self.model = self._prepareModel(analysisI)
        self.playerData = analysisI.getPlayerData()
    

    def _prepareModel(self,analysisI):
        #Dummy model for Now
        model = pickle.load(open(DB_PATH+'model.pkl', 'rb'))
        # model = DecisionTreeClassifier()
        # df = analysisI.create_data()
        # Y = df['result']
        # X = df.drop(columns=['result'])  
        # model.fit(X,Y)
        # pickle.dump(model, open('model.pkl', 'wb'))
        return model

    def predict(self,P1Info,P2Info,Surface,bestOf,indoor):
        df = pd.DataFrame(columns=['surface', 'best_of', 'indoor', 'elo_rating', 'rank', 'age', 'height', 'surface_win_p', 'indoor_p'],index=range(1,2))
        ID1 = P1Info['ID']
        ID2 = P2Info['ID']
        if Surface == "H":
            surface_win_p = self.playerData.loc[self.playerData['player_id'] == ID1, "hard_matches_won"].iloc[0] / self.playerData.loc[self.playerData['player_id'] == ID2, "hard_matches_won"].iloc[0]
        elif Surface == "G":
            surface_win_p = self.playerData.loc[self.playerData['player_id'] == ID1, "grass_matches_won"].iloc[0] / self.playerData.loc[self.playerData['player_id'] == ID2, "grass_matches_won"].iloc[0]
        elif Surface == "C":
            surface_win_p = self.playerData.loc[self.playerData['player_id'] == ID1, "clay_matches_won"].iloc[0] / self.playerData.loc[self.playerData['player_id'] == ID2, "clay_matches_won"].iloc[0]
        
        if indoor == 1:
            indoor_p = self.playerData.loc[self.playerData['player_id'] == ID1, "indoor_matches_won"].iloc[0] / self.playerData.loc[self.playerData['player_id'] == ID2, "indoor_matches_won"].iloc[0]
        elif indoor == 0:
            indoor_p = self.playerData.loc[self.playerData['player_id'] == ID1, "outdoor_matches_won"].iloc[0] / self.playerData.loc[self.playerData['player_id'] == ID2, "outdoor_matches_won"].iloc[0]
      
        age = P1Info['age'] /P2Info['age']
        rank = P1Info['rank']/P2Info['rank']
        height = P1Info['height']/P2Info['height']
        elo = P1Info['elo']/P2Info['elo']
        df.loc[len(df.index)] = [Surface,bestOf,indoor,elo,rank,age,height,surface_win_p,indoor_p]
        newVals = {
            "H":0,
            "C":1,
            "G":2,
        }

        df['surface'] = df['surface'].map(newVals)
        prediction = self.model.predict(df)[0]
        if prediction == 1:
            return ID1
        else:
            return ID2
    

if __name__ == "__main__":
    # a = Analysis()
    model = MLModel()
    P1Data = {
        "ID":5,
        "age":30,
        "height":170,
        "rank":10,
        "elo":1500        
    }

    P2Data = {
        "ID":7,
        "age":32,
        "height":180,
        "rank":5,
        "elo":1760        
    }
    surface = "H"
    bestof = 3
    indoor = 1
    WinnerID = model.predict(P1Data,P2Data,surface,bestof,indoor)
    print("Winner is Player With ID: {}".format(WinnerID))