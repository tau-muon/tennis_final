import pandas as pd
import numpy as np

url = "https://www.tennisexplorer.com/ranking/atp-men/?page="
n_pages = 40

mapper = {'Schwartzman Diego': 'Diego Sebastian Schwartzman',
          'Ramos-Vinolas Albert' : 'Albert Ramos',
          'Bautista-Agut Roberto' : 'Roberto Bautista Agut',
          'Struff Jan-Lennard' : 'Jan Lennard Struff',
          'Carreno-Busta Pablo' : 'Pablo Carreno Busta',
          'Garin Cristian' : 'Christian Garin',
          'McDonald Mackenzie' : 'Mackenzie Mcdonald',
          'Martinez Pedro' : 'Pedro Martinez Portero',
          'Harris Lloyd' : 'Lloyd George Muirhead Harris',
          'Alcaraz Carlos' : 'Carlos Alcaraz Garfia'
         }

def read_page(url, page):
    lst = pd.read_html(url + str(page))
    return lst[1]

def gen_ranking(url, n_pages):
    df = pd.DataFrame(columns=["Rank", "Move", "Player name", "Country", "Points"])
    
    for i in range(1, n_pages+1):
        df_page = read_page(url, i)    
        df = pd.concat([df, df_page], ignore_index=True)
    return df

def gen_id(df):
    res = df.copy()
    res['id'] = ["".join(sorted(x)) for x in res["Player name"].str.split().to_list()]
    return res

def map_names(df):
    res = df.copy()
    res['Player name'] = res['Player name'].map(mapper).fillna(res['Player name'])
    return res

def read_df(path):
    orig = pd.read_csv(path)
    orig = orig[orig["active"] == 't']
    orig = orig[orig['name'].notna()]
    orig['id'] = ["".join(sorted(x)) for x in orig['name'].str.split().to_list()]
    return orig

def join_df(df1, df2):
    res = df1.merge(df2, on = 'id', how = 'left')
    return res

def main(url, n_pages, orig_path):
    rankings = gen_id(map_names(gen_ranking(url, n_pages)))
    orig = read_df(orig_path)
    res = join_df(orig, rankings)
    res.to_csv("Player_info.csv", index=False)

if __name__ == "main":
    main(url, n_pages, "tennis_players_data.csv")
