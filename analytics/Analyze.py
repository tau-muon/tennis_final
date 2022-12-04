import os, sys
import pandas as pd
import numpy as np
import random


sys.path.append(os.path.abspath("../"))
from analytics.FeatEng import FeaturesEngineering


SEED = 903647789
random.seed(SEED)

class Analysis(object):
    """ Class created to perform the necessary data analysis and cleaning
    """

    def __init__(self,mode="eval") -> None:
        self.fe = FeaturesEngineering(mode=mode)

        self.match_df = self.clean()
        self.clean_data = None


    def clean_nanstats_match(self) -> pd.DataFrame:
        """ Clean the matches that doesn't have enough data with NA stats

        Returns:
            pd.DataFrame: Matches clean dataframe table
        """
        # Match column names
        #Index(['match_id', 'tournament_event_id', 'match_num', 'date', 'surface',
        #    'indoor', 'round', 'best_of', 'winner_id', 'winner_country_id',
        #    'winner_seed', 'winner_entry', 'winner_rank', 'winner_rank_points',
        #    'winner_elo_rating', 'winner_next_elo_rating', 'winner_age',
        #    'winner_height', 'loser_id', 'loser_country_id', 'loser_seed',
        #    'loser_entry', 'loser_rank', 'loser_rank_points', 'loser_elo_rating',
        #    'loser_next_elo_rating', 'loser_age', 'loser_height', 'score',
        #    'outcome', 'w_sets', 'l_sets', 'w_games', 'l_games', 'w_tbs', 'l_tbs',
        #    'has_stats'],
        #   dtype='object')
        df = self.fe.match_df
        df = df[df["winner_elo_rating"].notna()]
        df = df[df["loser_elo_rating"].notna()]
        df = df[df["winner_rank"].notna()]
        df = df[df["loser_rank"].notna()]
        df = df[df["winner_age"].notna()]
        df = df[df["winner_height"].notna()]
        df = df[df["loser_age"].notna()]
        df = df[df["loser_height"].notna()]
        df = df[df["surface"].notna()]
        df = df[df["best_of"].notna()]
        df = df[df["indoor"].notna()]
        return df


    def clean(self) -> pd.DataFrame:
        """ Perform all the cleaning needed to create the data directly

        Returns:
            pd.DataFrame: Dataframe with all the data needed
        """
        df = self.clean_nanstats_match()
        df = df[df["surface"] != "P"]
        return df


    def create_data(self) -> pd.DataFrame:
        """ Create the final dataframe clean table with all features and results calculated

        Returns:
            pd.DataFrame: Dataframe with all the features ready for use in the Machine Learning model 
        """
        # Shuffle to get the data that belongs to the player 1 and player 2
        self.match_df = self.match_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        
        # Merge the data frame in order to get the loser and winner detailed stats
        playerData = self.getPlayerData()
        merged_df = pd.merge(self.match_df, playerData, left_on="winner_id", right_on="player_id")
        merged_df = pd.merge(merged_df, playerData, left_on="loser_id", right_on="player_id", suffixes=("_winner", "_loser"))
  
        rowcutoff = int(merged_df.shape[0]/2)
        p1_data = merged_df[:rowcutoff]
        p2_data = merged_df[rowcutoff:]
    
        # Build p1 data
        df1 = pd.DataFrame()
        df1["surface"] = p1_data["surface"]
        df1["best_of"] = p1_data["best_of"]
        df1["indoor"] = p1_data["indoor"]

        df1["elo_rating"] = p1_data["winner_elo_rating"]/ p1_data["loser_elo_rating"]
        df1["rank"] = p1_data["winner_rank"]/p1_data["loser_rank"]

        df1["age"] = p1_data["winner_age"] / p1_data["loser_age"]
        df1["height"] = p1_data["winner_height"] / p1_data["loser_height"]

        df1.loc[df1["surface"] == "H", "surface_win_p"] = p1_data["hard_matches_won_p_winner"] / p1_data["hard_matches_won_p_loser"]
        df1.loc[df1["surface"] == "G", "surface_win_p"] = p1_data["grass_matches_won_p_winner"] / p1_data["grass_matches_won_p_loser"]
        df1.loc[df1["surface"] == "C", "surface_win_p"] = p1_data["clay_matches_won_p_winner"] / p1_data["clay_matches_won_p_loser"]
        df1.loc[df1["indoor"] == True, "indoor_p"] = p1_data["indoor_matches_won_p_winner"] / p1_data["indoor_matches_won_p_loser"]
        df1.loc[df1["indoor"] == False, "indoor_p"] = p1_data["outdoor_matches_won_p_winner"] / p1_data["outdoor_matches_won_p_loser"]
        df1.loc[df1["best_of"] == 3, "best_of_win_p"] = p1_data["best_of_3_matches_won_p_winner"] / p1_data["best_of_3_matches_won_p_loser"]
        df1.loc[df1["best_of"] == 5, "best_of_win_p"] = p1_data["best_of_5_matches_won_p_winner"] / p1_data["best_of_5_matches_won_p_loser"]

        df1["matches_won_p"] = p1_data["matches_won_p_winner"] / p1_data["matches_won_p_loser"]
        df1["backhand"] = p1_data["backhand_winner"] / p1_data["backhand_loser"]

        df1["result"] = 1
        
        # Build p2 data
        df2 = pd.DataFrame()
        df2["surface"] = p2_data["surface"]
        df2["best_of"] = p2_data["best_of"]
        df2["indoor"] = p2_data["indoor"]

        df2["elo_rating"] = p2_data["loser_elo_rating"] / p2_data["winner_elo_rating"]
        df2["rank"] = p2_data["loser_rank"] / p2_data["winner_rank"]

        df2["age"] = p2_data["loser_age"] / p2_data["winner_age"]
        df2["height"] =  p2_data["loser_height"] / p2_data["winner_height"]

        df2.loc[df2["surface"] == "H", "surface_win_p"] = p2_data["hard_matches_won_p_loser"] / p2_data["hard_matches_won_p_winner"]
        df2.loc[df2["surface"] == "G", "surface_win_p"] = p2_data["grass_matches_won_p_loser"] / p2_data["grass_matches_won_p_winner"]
        df2.loc[df2["surface"] == "C", "surface_win_p"] = p2_data["clay_matches_won_p_loser"] / p2_data["clay_matches_won_p_winner"]
        df2.loc[df2["indoor"] == True, "indoor_p"] = p2_data["indoor_matches_won_p_loser"] / p2_data["indoor_matches_won_p_winner"]
        df2.loc[df2["indoor"] == False, "indoor_p"] = p2_data["outdoor_matches_won_p_loser"] / p2_data["outdoor_matches_won_p_winner"]
        df2.loc[df2["best_of"] == 3, "best_of_win_p"] = p2_data["best_of_3_matches_won_p_loser"] / p2_data["best_of_3_matches_won_p_winner"]
        df2.loc[df2["best_of"] == 5, "best_of_win_p"] = p2_data["best_of_5_matches_won_p_loser"] / p2_data["best_of_5_matches_won_p_winner"]

        df2["matches_won_p"] = p2_data["matches_won_p_loser"] / p2_data["matches_won_p_winner"]
        df2["backhand"] = p2_data["backhand_loser"] / p2_data["backhand_winner"]

        df2["result"] = 2

        df = pd.concat([df1, df2])
        newVals = {
            "H":0,
            "C":1,
            "G":2,
        }

        df['surface'] = df['surface'].map(newVals)
        df["result"] = df["result"].astype("int")
        df["indoor"] = df["indoor"].map({False:0,True:1})

        print(df.columns.tolist(), len(df.columns.tolist()))
        return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


    def getPlayerData(self):
        df = self.fe.player_performance_df
        for surface in ["clay_", "grass_", "hard_","indoor_","outdoor_","", "best_of_3_", "best_of_5_"]:
            df[surface+"matches_won_p"] = df[surface+"matches_won"] / (df[surface+"matches_won"] + df[surface+"matches_lost"])
            df[surface+"matches_won_p"] = df[surface+"matches_won_p"].replace(np.nan, 0.5)
            df.loc[df[surface+"matches_won_p"] == 0, surface+"matches_won_p"] = 0.5

        playerData_df = pd.merge(self.fe.player_df, df, left_on="player_id", right_on="player_id")

        playerData_df = playerData_df[playerData_df["active"] == True]
        playerData_df = playerData_df[playerData_df["backhand"].notna()]
        playerData_df["backhand"] = playerData_df["backhand"].apply(pd.to_numeric)

        # playerData_df.to_csv("./db/all_player_data.csv", index=False)
        return playerData_df


if __name__ == "__main__":
    a = Analysis(mode="prod")
    a.create_data()