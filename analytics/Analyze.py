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

        self.match_df = self.clean_nanstats_match()
        self.original_df = self.create_data()
        self.clean_data = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None


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
        return df


    def clean_retirement_matches(self):
        return


    def clean_minimum_match_count(self):
        return


    def clean_2020_matchs(self):
        return


    def create_data(self):
        # TODO to check the draw data
    
        # Shuffle to get the data that belongs to the player 1 and player 2
        self.match_df = self.match_df.sample(frac=1).reset_index(drop=True)
        
        merged_df = pd.merge(self.match_df, self.fe.player_df, left_on="winner_id", right_on="player_id")
        merged_df = pd.merge(merged_df, self.fe.player_df, left_on="loser_id", right_on="player_id", suffixes=("_winner", "_loser"))
        merged_df.to_csv("full_data.csv")

        rowcutoff = int(merged_df.shape[0]/2)
        p1_data = merged_df[:rowcutoff]
        p2_data = merged_df[rowcutoff:]
    
        # Build p1 data
        df1 = pd.DataFrame()
        df1["match_id"] = p1_data["match_id"]
        df1["surface"] = p1_data["surface"]
        df1["best_of"] = p1_data["best_of"]
        df1["indoor"] = p1_data["indoor"]

        df1["elo_rating"] = abs((p1_data["winner_elo_rating"] - p1_data["loser_elo_rating"]) / (p1_data["winner_elo_rating"] + p1_data["loser_elo_rating"]))
        df1["rank"] = abs((p1_data["winner_rank"] - p1_data["loser_rank"])/ (p1_data["winner_rank"] + p1_data["loser_rank"])) 

        df1.loc[p1_data["winner_age"] > p1_data["loser_age"], "age"] = 1
        df1.loc[p1_data["winner_age"] < p1_data["loser_age"], "age"] = 2
        df1.loc[p1_data["winner_age"] == p1_data["loser_age"], "age"] = 0

        df1.loc[p1_data["winner_height"] > p1_data["loser_height"], "height"] = 1
        df1.loc[p1_data["winner_height"] < p1_data["loser_height"], "height"] = 2
        df1.loc[p1_data["winner_height"] == p1_data["loser_height"], "height"] = 0
        df1["result"] = 1
        
        # Build p2 data
        df2 = pd.DataFrame()
        df2["match_id"] = p2_data["match_id"]
        df2["surface"] = p2_data["surface"]
        df2["best_of"] = p2_data["best_of"]
        df2["indoor"] = p2_data["indoor"]
        df2["elo_rating"] = abs((p2_data["winner_elo_rating"] - p2_data["loser_elo_rating"]) / (p2_data["winner_elo_rating"] + p2_data["loser_elo_rating"]))
        df2["rank"] = abs((p2_data["winner_rank"] - p2_data["loser_rank"])/ (p2_data["winner_rank"] + p2_data["loser_rank"])) 
        df2.loc[p2_data["winner_age"] > p2_data["loser_age"], "age"] = 1
        df2.loc[p2_data["winner_age"] < p2_data["loser_age"], "age"] = 2
        df2.loc[p2_data["winner_age"] == p2_data["loser_age"], "age"] = 0
        df2.loc[p2_data["winner_height"] > p2_data["loser_height"], "height"] = 1
        df2.loc[p2_data["winner_height"] < p2_data["loser_height"], "height"] = 2
        df2.loc[p2_data["winner_height"] == p2_data["loser_height"], "height"] = 0
        df2["result"] = 2

        df = pd.concat([df1, df2])
        return df.sample(frac=1).reset_index(drop=True)


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


if __name__ == "__main__":
    a = Analysis()
    a.create_data()
    a.original_df.to_csv("origina_data.csv")