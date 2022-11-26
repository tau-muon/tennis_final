import pandas as pd
import requests
import json


#The functions below are intended to interact with http://www.tennis-data.co.uk/

def read_data(path):
    res = pd.read_csv(path)
    return res

def winning_rate(player):
    wins = len(df[df["Winner"] == player])
    loss = len(df[df["Loser"] == player])
    return wins / (wins + loss)


def winning_rate_by_surface(player, surface):
    wins = len(df[(df["Winner"] == player) & (df["Surface"] == surface)])
    loss = len(df[(df["Loser"] == player) & (df["Surface"] == surface)])
    try:
        res = wins / (wins + loss)
    except ZeroDivisionError:
        res = "No matches on this court"
    return res

def wr_over_time(player):
    wins = df[df["Winner"] == player].groupby(df["Date"].dt.month).size()
    loss = df[df["Loser"] == player].groupby(df["Date"].dt.month).size()
    wins.name = "Wins"
    loss.name = "Loss"
    res = pd.DataFrame(wins).join(loss, how = 'outer')
    res = res.reset_index()
    res["win_rate"] = res["Wins"] / (res["Wins"] + res["Loss"])
    return res

#The functions below are intended to interact with https://api-tennis.com/

api_key = "ee19c554e4b555e7508cb151dc74677520c82f290845cfc9f63cbec685fef8e1"

def get_head_to_head(api_key, p1_key, p2_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_H2H&APIkey={api_key}&first_player_key={p1_key}&second_player_key={p2_key}"
    response = requests.get(url)
    return json.loads(response.text)['result']

def get_standings(api_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_standings&event_type=ATP&APIkey={api_key}"
    response = requests.get(url)
    return json.loads(response.text)['result']

def get_player(api_key, p_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_players&player_key={p_key}&APIkey={api_key}"
    response = requests.get(url)
    return json.loads(response.text)['result']

def json_to_df(jsondata):
    return pd.DataFrame(jsondata)

def player_key(player_name, df):
    return int(df[df['player'] == player_name]['player_key'])

def list_of_players(df):
    return list(df['player'])

def current_ranking(p_key, df):
    return int(df[df['player_key'] == str(p_key)]['place'])
