import os, sys

from Database import Database


# Tournament:
# #Surface
# #Indoor Vs Outdoor
# #Best of 3 or Best of 5
# Player:
# Hand -- Replace with P(Hand == Winner) -- Important: Must only be checked when players are not both right handed or left-handed. 0 1 2 3
# height -- 0 1 2 3
# weight -- 0 1 2 3
# backhand -- One handed backhand or two handed backhand -- Must only be checked when players are not both right handed or left-handed. 0 1 2 3
# birthplace --
# Current Ranking Official -- R1-R2
# Current ELO Ranking -- R1-R2
# Historic BO3,BO5 Match Percent Win under Format
# Historic Surface Match Percent win
# Current Season BO3,BO5 Match Percent Win  
# Current Season Surface Match Percent win
# This Tournament Match Percent Win
# PlayerVPlayer
# Head-to-head
# Head-to-Head on Target Surface
# Head-to-head on Target Tournament

class FeaturesEngineering(object):
    def __init__(self) -> None:
        self.db = Database()
    
    def get_active_player_table(self):
        columns = ["player_id", "first_name", "last_name", "height", "weight", "hand", "backhand", "active", "turned_pro" , "birthplace"]
        df = self.db.getdf("player")[columns]
        return df[df["active"] == True]
    
    def get_player_performance_table(self):
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
        return self.db.getdf("player_performance")[columns]
    
    def gen_csv(self):
        """ Return a csv that will be used for machine learning module
        """
        return


if __name__ == "__main__":
    feat = FeaturesEngineering()
    print(feat.get_active_player_table())
    print(feat.get_player_performance_table().columns)

