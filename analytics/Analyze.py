import os, sys

sys.path.append(os.path.abspath("../"))

from analytics.FeatEng import FeaturesEngineering

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

class Analyze(object):
    """ Class created to perform the necessary data analysis and cleaning
    """

    def __init__(self) -> None:
        pass


    def clean(self):
        return


    def dumpcsv(self):
        return
