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

    def __init__(self) -> None:
        self.fe = FeaturesEngineering()

        self.match_df = self.clean()
        # self.original_df = self.create_data()
        self.clean_data = None


    def clean_nanstats_match(self) -> pd.DataFrame:
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


    def clean_retirement_matches(self):
        return


    def clean_minimum_match_count(self, match_count:int):
        return


    def clean_2020_matchs(self, df:pd.DataFrame):
        # print(df["date"])
        return df


    def clean(self) -> pd.DataFrame:
        """ Perform all the cleaning needed to create the data directly

        Returns:
            pd.DataFrame: Dataframe with all the data needed
        """
        df = self.clean_nanstats_match()
        df = self.clean_2020_matchs(df)

        df = df[df["surface"] != "P"]
        return df


    def create_data(self):
        # Shuffle to get the data that belongs to the player 1 and player 2
        self.match_df = self.match_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        
        merged_df = pd.merge(self.match_df, self.fe.player_performance_df, left_on="winner_id", right_on="player_id")
        merged_df = pd.merge(merged_df, self.fe.player_performance_df, left_on="loser_id", right_on="player_id", suffixes=("_winner", "_loser"))
        merged_df.to_csv("full_data.csv")
        merged_df["result"] = -1

        player_id_list = merged_df["winner_id"].tolist()
        player_id_list.extend(merged_df["loser_id"].tolist())
        player_id_list = list(set(player_id_list))
        # print("Loop on player id:")
        for player_id in player_id_list:
            player_matchs_df = merged_df[(merged_df["winner_id"] == player_id) | (merged_df["loser_id"] == player_id)].sample(frac=1, random_state=SEED).reset_index(drop=True)
            rowcutoff = int(player_matchs_df.shape[0]/2)
            merged_df.loc[merged_df.index.isin(player_matchs_df.index[:rowcutoff]) & merged_df["result"] == -1, "result"] = 1
            merged_df.loc[merged_df.index.isin(player_matchs_df.index[rowcutoff:]) & merged_df["result"] == -1, "result"] = 2

        # print(merged_df.shape)
        # print(merged_df["result"].to_string())
        # print(merged_df[merged_df["result"] == -1].shape)

        print(merged_df.columns.tolist())

        merged_df["clay_matches_won_winner"] = merged_df["clay_matches_won_winner"].replace(np.nan, 0.5)
        merged_df.loc[merged_df["clay_matches_won_winner"] == 0, "clay_matches_won_winner"] = 0.5
        merged_df["clay_matches_won_loser"] = merged_df["clay_matches_won_loser"].replace(np.nan, 0.5)
        merged_df.loc[merged_df["clay_matches_won_loser"] == 0, "clay_matches_won_loser"] = 0.5

        merged_df["grass_matches_won_winner"] = merged_df["grass_matches_won_winner"].replace(np.nan, 0.5)
        merged_df.loc[merged_df["grass_matches_won_winner"] == 0, "grass_matches_won_winner"] = 0.5
        merged_df["grass_matches_won_loser"] = merged_df["grass_matches_won_loser"].replace(np.nan, 0.5)
        merged_df.loc[merged_df["grass_matches_won_loser"] == 0, "grass_matches_won_loser"] = 0.5

        merged_df["hard_matches_won_winner"] = merged_df["hard_matches_won_winner"].replace(np.nan, 0.5)
        merged_df.loc[merged_df["hard_matches_won_winner"] == 0, "hard_matches_won_winner"] = 0.5
        merged_df["hard_matches_won_loser"] = merged_df["hard_matches_won_loser"].replace(np.nan, 0.5)
        merged_df.loc[merged_df["hard_matches_won_loser"] == 0, "hard_matches_won_loser"] = 0.5

        merged_df["indoor_matches_won_winner"] = merged_df["indoor_matches_won_winner"].replace(np.nan, 0.5)
        merged_df.loc[merged_df["indoor_matches_won_winner"] == 0, "indoor_matches_won_winner"] = 0.5
        merged_df["indoor_matches_won_loser"] = merged_df["indoor_matches_won_loser"].replace(np.nan, 0.5)
        merged_df.loc[merged_df["indoor_matches_won_loser"] == 0, "indoor_matches_won_loser"] = 0.5

        merged_df["outdoor_matches_won_winner"] = merged_df["outdoor_matches_won_winner"].replace(np.nan, 0.5)
        merged_df.loc[merged_df["outdoor_matches_won_winner"] == 0, "outdoor_matches_won_winner"] = 0.5
        merged_df["outdoor_matches_won_loser"] = merged_df["outdoor_matches_won_loser"].replace(np.nan, 0.5)
        merged_df.loc[merged_df["outdoor_matches_won_loser"] == 0, "outdoor_matches_won_loser"] = 0.5

        rowcutoff = int(merged_df.shape[0]/2)
        # rowcutoff = 1
        p1_data = merged_df[:rowcutoff]
        p2_data = merged_df[rowcutoff:]
    
        # Build p1 data
        df1 = pd.DataFrame()
        # df1["match_id"] = p1_data["match_id"]
        df1["surface"] = p1_data["surface"]
        df1["best_of"] = p1_data["best_of"]
        df1["indoor"] = p1_data["indoor"]

        # df1["elo_rating"] = (p1_data["winner_elo_rating"] - p1_data["loser_elo_rating"]) / (p1_data["winner_elo_rating"] + p1_data["loser_elo_rating"])
        # df1["rank"] = (p1_data["winner_rank"] - p1_data["loser_rank"])/ (p1_data["winner_rank"] + p1_data["loser_rank"])
        df1["elo_rating"] = (p1_data["winner_elo_rating"]/ p1_data["loser_elo_rating"])
        df1["rank"] = (p1_data["winner_rank"]/p1_data["loser_rank"])

        # df1.loc[p1_data["winner_age"] > p1_data["loser_age"], "age"] = 1
        # df1.loc[p1_data["winner_age"] < p1_data["loser_age"], "age"] = 2
        # df1.loc[p1_data["winner_age"] == p1_data["loser_age"], "age"] = 0
        df1["age"] = p1_data["winner_age"] / p1_data["loser_age"]

        # df1.loc[p1_data["winner_height"] > p1_data["loser_height"], "height"] = 1
        # df1.loc[p1_data["winner_height"] < p1_data["loser_height"], "height"] = 2
        # df1.loc[p1_data["winner_height"] == p1_data["loser_height"], "height"] = 0
        df1["height"] = p1_data["winner_height"] / p1_data["loser_height"]

        # df1.loc[df1["surface"] == "P", "surface_win_p"] = p1_data["carpet_matches_won_winner"] / p1_data["carpet_matches_won_loser"]
        df1.loc[df1["surface"] == "H", "surface_win_p"] = p1_data["hard_matches_won_winner"] / p1_data["hard_matches_won_loser"]
        df1.loc[df1["surface"] == "G", "surface_win_p"] = p1_data["grass_matches_won_winner"] / p1_data["grass_matches_won_loser"]
        df1.loc[df1["surface"] == "C", "surface_win_p"] = p1_data["clay_matches_won_winner"] / p1_data["clay_matches_won_loser"]

        df1.loc[df1["indoor"] == True, "indoor_p"] = p1_data["indoor_matches_won_winner"] / p1_data["indoor_matches_won_loser"]
        df1.loc[df1["indoor"] == False, "indoor_p"] = p1_data["outdoor_matches_won_winner"] / p1_data["outdoor_matches_won_loser"]

        df1["result"] = 1
        # df1["result"] = p1_data["result"]
        
        # Build p2 data
        df2 = pd.DataFrame()
        # df2["match_id"] = p2_data["match_id"]
        df2["surface"] = p2_data["surface"]
        df2["best_of"] = p2_data["best_of"]
        df2["indoor"] = p2_data["indoor"]

        df2["elo_rating"] = (p2_data["loser_elo_rating"] / p2_data["winner_elo_rating"])
        df2["rank"] = (p2_data["loser_rank"] / p2_data["winner_rank"])

        # df2.loc[p2_data["winner_age"] < p2_data["loser_age"], "age"] = 1
        # df2.loc[p2_data["winner_age"] > p2_data["loser_age"], "age"] = 2
        # df2.loc[p2_data["winner_age"] == p2_data["loser_age"], "age"] = 0
        df2["age"] = p2_data["loser_age"] / p2_data["winner_age"]

        # df2.loc[p2_data["winner_height"] < p2_data["loser_height"], "height"] = 1
        # df2.loc[p2_data["winner_height"] > p2_data["loser_height"], "height"] = 2
        # df2.loc[p2_data["winner_height"] == p2_data["loser_height"], "height"] = 0
        df2["height"] =  p2_data["loser_height"] / p2_data["winner_height"]


        # df2.loc[df2["surface"] == "P", "surface_win_p"] = p2_data["carpet_matches_won_loser"] / p2_data["carpet_matches_won_winner"]
        df2.loc[df2["surface"] == "H", "surface_win_p"] = p2_data["hard_matches_won_loser"] / p2_data["hard_matches_won_winner"]
        df2.loc[df2["surface"] == "G", "surface_win_p"] = p2_data["grass_matches_won_loser"] / p2_data["grass_matches_won_winner"]
        df2.loc[df2["surface"] == "C", "surface_win_p"] = p2_data["clay_matches_won_loser"] / p2_data["clay_matches_won_winner"]
        
        df2.loc[df2["indoor"] == True, "indoor_p"] = p2_data["indoor_matches_won_loser"] / p2_data["indoor_matches_won_winner"]
        df2.loc[df2["indoor"] == False, "indoor_p"] = p2_data["outdoor_matches_won_loser"] / p2_data["outdoor_matches_won_winner"]

        df2["result"] = 2
        # df2["result"] = p2_data["result"]

        df = pd.concat([df1, df2])
        newVals = {
            "H":0,
            "C":1,
            "G":2,
        }

        df['surface'] = df['surface'].map(newVals)
        df["result"] = df["result"].astype("int")
        df["indoor"] = df["indoor"].map({False:0,True:1})

        print(df.columns.tolist())
        return df.sample(frac=1).reset_index(drop=True)

        # #### Looking at the problem from player 1 and player 2 angle
        # match_df = self.clean()
        # print(match_df.columns)
        # players_df = self.fe.player_performance_df
        # print(players_df.columns)

        # players_len = players_df.shape[0]
        # df = pd.DataFrame(index=range(0, players_len**2), columns=["player_1_id", "player_2_id"])
        # df["player_1_id"] = players_df["player_id"].tolist()*players_len
        # i = 0
        # max_index = players_len**2
        # while i <= max_index:
        #     player_index = i % players_len
        #     df["player_2_id"].iloc[i: i+players_len+1] = [players_df["player_id"].iloc[player_index]]*players_len
        #     i += players_len
        # return df


    def map_x(self):
        # X data are the match features as filtered and extracted from the feature engineering part
        self.fe.match_df
        return


    def map_y(self):
        # y values are either 0, 1, 2 
        # 0 for draw match result
        # 1 for player 1 is a winner
        # 2 for player 2 is a winner
        self.fe.match_df
        return


    def dumpcsv(self):
        return


# if __name__ == "__main__":
#     a = Analysis()
#     a.create_data()
#     a.original_df.to_csv("origina_data.csv")