import pandas as pd

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

