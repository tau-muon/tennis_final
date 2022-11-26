import os, sys
import pandas as pd
import subprocess

sys.path.append(os.path.abspath("../"))

from analytics.Database import Database


class FeaturesEngineering(object):
    def __init__(self) -> None:
        self.db = Database()
        # Player data
        self.player_df = self.get_active_player_table()
        self.player_performance_df = self.get_player_performance_table()
        self.player_stats_df = self.get_player_stats_table()
        self.player_h2h_df = self.get_player_h2h_table()

        # Match data
        self.match_df = self.get_match_table()
        self.tournament_df = self.get_tournament_table()
        

    def get_active_player_table(self) -> pd.DataFrame:
        """ Extract the player table with needed columns and players being active

        Returns:
            pd.DataFrame: Active player dataframe.
        """
        drop_columns = ["twitter", "residence", "wikipedia", "web_site", "short_name", "facebook"]
        df = self.db.getdf(tableName="player_v")
        df = df.drop(columns=drop_columns)
        # ['player_id', 'first_name', 'last_name', 'dob', 'dod', 'country_id',
        # 'birthplace', 'height', 'weight', 'hand', 'backhand', 'active',
        # 'turned_pro', 'coach', 'prize_money', 'nicknames', 'name',
        # 'age', 'current_rank', 'current_rank_points', 'best_rank',
        # 'best_rank_date', 'best_rank_points', 'best_rank_points_date',
        # 'current_elo_rank', 'current_elo_rating', 'best_elo_rank',
        # 'best_elo_rank_date', 'best_elo_rating', 'best_elo_rating_date',
        # 'goat_rank', 'goat_points', 'weeks_at_no1', 'titles', 'big_titles',
        # 'grand_slams', 'tour_finals', 'alt_finals', 'masters', 'olympics']
        return df[df["active"] == True]

    
    def get_player_performance_table(self) -> pd.DataFrame:
        """ Extract the player performance table with features needed to perform analytics needed.

        Returns:
            pd.DataFrame: PLayer performance dataframe.
        """
        columns = ['player_id', 'matches_won', 'matches_lost', 'grand_slam_matches_won',
                    'grand_slam_matches_lost', 'tour_finals_matches_won',
                    'tour_finals_matches_lost', 'alt_finals_matches_won',
                    'alt_finals_matches_lost', 'masters_matches_won',
                    'masters_matches_lost', 'olympics_matches_won', 'olympics_matches_lost',
                    'atp500_matches_won', 'atp500_matches_lost', 'atp250_matches_won',
                    'atp250_matches_lost', 'best_of_3_matches_won',
                    'best_of_3_matches_lost', 'best_of_5_matches_won',
                    'best_of_5_matches_lost', 'hard_matches_won', 'hard_matches_lost',
                    'clay_matches_won', 'clay_matches_lost', 'grass_matches_won',
                    'grass_matches_lost', 'carpet_matches_won', 'carpet_matches_lost',
                    'outdoor_matches_won', 'outdoor_matches_lost', 'indoor_matches_won',
                    'indoor_matches_lost', 'deciding_sets_won', 'deciding_sets_lost',
                    'fifth_sets_won', 'fifth_sets_lost', 'finals_won', 'finals_lost']
        return self.db.getdf(tableName="player_performance")[columns]


    def get_player_stats_table(self) -> pd.DataFrame:
        return self.db.getdf(tableName="player_stats_v")


    def get_player_h2h_table(self) -> pd.DataFrame:
        return self.db.getdf(tableName="player_h2h")


    def get_tournament_table(self) -> pd.DataFrame:
        # ['tournament_id', 'name', 'country_id', 'city', 'level', 'surface', 'indoor', 'linked']
        return self.db.getdf(tableName="tournament")


    def get_match_table(self) -> pd.DataFrame:
        return self.db.getdf(tableName="match")

    
    def dump_tableNames(self):
        self.db.getdf_tablenames()[["table_catalog","table_schema","table_name","table_type"]].to_csv("tableNames.csv")


if __name__ == "__main__":
    output = subprocess.run(["sudo","docker", "start" ,"uts-database"], stdout=subprocess.PIPE)
    if output.returncode != 0:
        print(output.stdout.decode("utf-8"))
        sys.exit("Failed to launch database docker!")
    feat = FeaturesEngineering()

