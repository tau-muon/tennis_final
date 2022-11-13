# Analysis Module

## Introduction

Extracting the features to be used for match prediction requires looking through all the tables we have already first.

Player:

- Hand -- Replace with P(Hand == Winner) -- Important: Must only be checked when players are not both right handed or left-handed. 0 1 2 3
- height -- 0 1 2 3
- weight -- 0 1 2 3
- backhand -- One handed backhand or two handed backhand -- Must only be checked when players are not both right handed or left-handed. 0 1 2 3
- Current Ranking Official -- R1-R2
- Current ELO Ranking -- R1-R2
- Historic BO3,BO5 Match Percent Win under Format
- Historic Surface Match Percent win
- Current Season BO3,BO5 Match Percent Win  
- Current Season Surface Match Percent win

This Tournament Match Percent Win

- PlayerVPlayer
- Head-to-head
- Head-to-Head on Target Surface
- Head-to-head on Target Tournament
