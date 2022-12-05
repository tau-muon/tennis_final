import sklearn as sk
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import sys, os
import pickle

sys.path.append(os.path.abspath("../"))
from analytics.FeatEng import DB_PATH
from analytics.Analyze import Analysis, SEED


class MLModel(object):
    def __init__(self) -> None:
        analysisI = Analysis(mode="prod")
        model = self._loadModel()
        if model == None:
            model = self._prepareModel(analysisI=analysisI)
        self.model = model
        self.playerData = analysisI.getPlayerData()
    

    def _loadModel(self):
        try:
            model = pickle.load(open(DB_PATH+'model.pkl', 'rb'))
        except:
            model = None
        return model

    def _prepareModel(self,analysisI:Analysis) -> object:
        """ Train and save the model in case first time running the system

        Args:
            analysisI (Analysis): Class group to get data needed for analysis

        Returns:
            object: Model trained on all the data currently being in use
        """
        model = GradientBoostingClassifier(n_estimators=512,random_state=SEED)
        df = analysisI.create_data()
        Y = df['result']
        X = df.drop(columns=['result'])  
        model.fit(X,Y)
        pickle.dump(model, open(DB_PATH+'model.pkl', 'wb'))
        return model

    def predict(self,ID1,ID2,Surface,bestOf,indoor):
        
        
        df = pd.DataFrame(columns=['surface', 'best_of', 'indoor', 'elo_rating', 'rank', 'age', 'height', 'surface_win_p', 'indoor_p', 'best_of_win_p', 'matches_won_p', 'backhand'],index=range(1,2))
        player1Data = self.playerData[self.playerData["player_id"] == ID1]
        player2Data = self.playerData[self.playerData["player_id"] == ID2]

        #### Calculate the surface winning percentage
        if Surface == "H":
            surface_win_p = player1Data["hard_matches_won_p"].iloc[0] / player2Data["hard_matches_won_p"].iloc[0]
        elif Surface == "G":
            surface_win_p = player1Data["grass_matches_won_p"].iloc[0] / player2Data["grass_matches_won_p"].iloc[0]
        elif Surface == "C":
            surface_win_p = player1Data["clay_matches_won_p"] / player2Data["clay_matches_won_p"].iloc[0]

        #### Calculate the indoor/outdoor winning percentage
        if indoor == 1:
            indoor_p = player1Data["indoor_matches_won_p"].iloc[0] / player2Data["indoor_matches_won_p"].iloc[0]
        elif indoor == 0:
            indoor_p = player1Data["outdoor_matches_won_p"].iloc[0] / player2Data["outdoor_matches_won_p"].iloc[0]
        
        #### Calculate the best of winning percentage
        if bestOf == 3:
            best_of_win_p = player1Data["best_of_3_matches_won_p"].iloc[0] / player2Data["best_of_3_matches_won_p"].iloc[0]
        elif bestOf == 5:
            best_of_win_p = player1Data["best_of_5_matches_won_p"].iloc[0] / player2Data["best_of_5_matches_won_p"].iloc[0]

        #### Calculate the overall winning percentage
        matches_win_p =  player1Data["matches_won_p"].iloc[0] / player2Data["matches_won_p"].iloc[0]

        #### Calculate backhand winning percentage
        backhand_matches_win_p = player1Data["backhand"].iloc[0] / player2Data["backhand"].iloc[0]

        age = player1Data['age'].iloc[0] /player2Data['age'].iloc[0]
        rank = player1Data['current_rank'].iloc[0] / player2Data['current_rank'].iloc[0]
        height = player1Data['height'].iloc[0] / player2Data['height'].iloc[0]
        elo = player1Data['current_elo_rating'].iloc[0] / player2Data['current_elo_rating'].iloc[0]
        if pd.isna(elo):
            elo = 1
        if pd.isna(rank):
            rank = 1
        if pd.isna(age):
            age =1
        if pd.isna(height):
            height = 1
        df.loc[len(df.index)] = [Surface,bestOf,indoor,elo,rank,age,height,surface_win_p,indoor_p, best_of_win_p, matches_win_p, backhand_matches_win_p]
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
    p1_id = 644
    p2_id = 5216
    surface = "H"
    bestof = 3
    indoor = 1
    best_of = 3
    WinnerID = model.predict(p1_id,p2_id,surface,bestof,indoor)
    print("Winner is Player With ID: {}".format(WinnerID))
    subprocess.call(["git", "commit", "-a", "-m", "adding pickle file"])
    
    
    
