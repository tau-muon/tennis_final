import pandas as pd

url = "https://www.tennisexplorer.com/ranking/atp-men/?page="
n_pages = 40

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
    res = df
    res['id'] = ["".join(sorted(x)) for x in res["Player name"].str.split().to_list()]
    return res