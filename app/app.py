# Import Libraries

import sys
import dash
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import requests
import json
from dash import html, dcc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

sys.path.insert(0, "../analytics")
from ML import MLModel

# Helper functions

def get_standings(api_key):
    url = f"https://api.api-tennis.com/tennis/?method=get_standings&event_type=ATP&APIkey={api_key}"
    response = requests.get(url)
    return json.loads(response.text)['result']

# Read the data

df = pd.read_csv("data/tennis_players_data.csv", index_col="player_id")
df = df[df.api_id != "NOID"]
df_radar = df[["name", "matches_win_percentage", "grand_slam_win_percentage", "tour_finals_win_percentage",
               "olympics_matches_win_percentage", "davis_cup_matches_win_percentage", "hard_matches_win_percentage",
               "clay_matches_win_percentage", "grass_matches_win_percentage", "carpet_matches_win_percentage",
               "outdoor_matches_win_percentage", "indoor_matches_win_percentage"]]

BASE_URL = 'https://api.api-tennis.com/tennis/?'
API_KEY = "84818ec2586012d56c6d009790776455e75e6d7f9221ad8ee70d40d2dec185ff"
method = 'method=get_players'


standings = pd.DataFrame(get_standings(API_KEY))[["player", "place", "points", 'player_key']]
model = MLModel()
# country_code = dict(df["country_id"])



# Function for figures

def radar_chart(data, player_id_1, player_id_2):
    df_graph = data[data.index.isin([player_id_1, player_id_2])]

    categories = ['Matches Won', 'Grand Slam Matches Won', 'Tour Finals Matches Won',
                  'Olympics Matches Won', 'Davis Cup Matches Won', 'Hard Matches Won',
                  'Clay Matches Won', 'Grass Matches Won', 'Carper Matches Won',
                  'Outdoor Matches Won', 'Indoor Matches Won']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(df_graph.iloc[0, 1:]),
        theta=categories,
        fill='toself',
        name=df_graph.iloc[0, 0]
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(df_graph.iloc[1, 1:]),
        theta=categories,
        fill='toself',
        name=df_graph.iloc[1, 0]
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,

    )

    fig.update_layout(title_text='', title_x=0.5, title_y=0.95, title_font_family="Old Standard TT",
                      title_font_size=40, title_font_color='green', paper_bgcolor='white', plot_bgcolor='red')

    return fig


def show_country(player_id_1, player_id_2):
    data = df
    player1_country_id = list(data[data.index.isin([player_id_1, player_id_2])]['country_id'])[0]
    player2_country_id = list(data[data.index.isin([player_id_1, player_id_2])]['country_id'])[1]
    gapminder = px.data.gapminder()
    data_file = gapminder[
        (gapminder['iso_alpha'] == player1_country_id) | (gapminder['iso_alpha'] == player2_country_id)]
    fig = px.choropleth(data_file, locations="iso_alpha",
                        color="country",  # lifeExp is a column of gapminder
                        hover_name="country",  # column to add to hover information
                        color_continuous_scale=px.colors.sequential.Plasma)
    fig.update_layout(title_text='', title_x=0.5, title_y=0.95, title_font_family="Old Standard TT",
                      title_font_size=40, title_font_color='green', paper_bgcolor='white', plot_bgcolor='red')
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        #paper_bgcolor="LightSteelBlue",
    )
    return fig


# ----------18th Nov 2022

def country_id(player_id):
    data = df
    country = list(data[data.index.isin([player_id])]['country_id_2'])[0]
    return country


def get_country_flag(player_id):
    country = country_id(player_id)
    flag = f"https://countryflagsapi.com/svg/{country}"
    return flag


def api_id(player_id):
    data = df
    id_api = list(data[data.index.isin([player_id])]['api_id'])[0]
    return id_api


def rank_season(player1_id, player2_id):
    api_p1 = api_id(player1_id)
    api_p2 = api_id(player2_id)

    complete_url_1 = f"{BASE_URL}{method}&player_key={api_p1}&APIkey={API_KEY}"
    complete_url_2 = f"{BASE_URL}{method}&player_key={api_p2}&APIkey={API_KEY}"
    response_1 = requests.get(complete_url_1)
    response_2 = requests.get(complete_url_2)
    player_1_data = response_1.json()
    player_2_data = response_2.json()

    player_1_df = pd.DataFrame(player_1_data['result'][0]['stats'])
    player_1_df = player_1_df[player_1_df['type'] == 'singles']
    player_1_df = player_1_df.replace('', 0)
    player_1_df[
        ['rank', 'titles', 'matches_won', 'matches_lost', 'hard_won', 'hard_lost', 'clay_won', 'clay_lost', 'grass_won',
         'grass_lost']] = player_1_df[
        ['rank', 'titles', 'matches_won', 'matches_lost', 'hard_won', 'hard_lost', 'clay_won', 'clay_lost', 'grass_won',
         'grass_lost']].astype(int)

    player_2_df = pd.DataFrame(player_2_data['result'][0]['stats'])
    player_2_df = player_2_df[player_2_df['type'] == 'singles']
    player_2_df = player_2_df.replace('', 0)
    player_2_df[
        ['rank', 'titles', 'matches_won', 'matches_lost', 'hard_won', 'hard_lost', 'clay_won', 'clay_lost', 'grass_won',
         'grass_lost']] = player_2_df[
        ['rank', 'titles', 'matches_won', 'matches_lost', 'hard_won', 'hard_lost', 'clay_won', 'clay_lost', 'grass_won',
         'grass_lost']].astype(int)

    df1 = player_1_df[['season', 'rank', 'titles']].iloc[::-1]
    df2 = player_2_df[['season', 'rank', 'titles']].iloc[::-1]
    df3 = df1.merge(df2, on='season', how='outer')
    df3.sort_values(by=["season"], inplace=True)
    df3.rename(columns={'rank_x': 'player_1_rank', 'titles_x': 'player_1_title', 'rank_y': 'player_2_rank',
                        'titles_y': 'player_2_title'}, inplace=True, errors='raise')
    df3.replace(to_replace=0, value=np.nan, inplace=True)

    fig = px.line(df3, x='season',
                  y=['player_1_rank', 'player_2_rank'],
                  # labels = {}
                  markers=True)

    # Title
    annotations = [dict(xref='paper', yref='paper', x=0.0, y=1.05,
                        xanchor='left', yanchor='bottom',
                        text=f"Player 1: {player_1_data['result'][0]['player_name']} vs Player 2:"
                             f" {player_2_data['result'][0]['player_name']}  Rank over Years",
                        font=dict(family='Arial',
                                  size=30,
                                  color='rgb(37,37,37)'),
                        showarrow=False), dict(xref='paper', yref='paper', x=0.5, y=-0.25,
                                               xanchor='center', yanchor='top',
                                               text='Source: https://api-tennis.com/',
                                               font=dict(family='Arial',
                                                         size=12,
                                                         color='rgb(150,150,150)'),
                                               showarrow=False)]
    # Source

    fig.update_layout(annotations=annotations)

    return fig


def title_season(player1_id, player2_id):
    complete_url_1 = f"{BASE_URL}{method}&player_key={api_id(player1_id)}&APIkey={API_KEY}"
    complete_url_2 = f"{BASE_URL}{method}&player_key={api_id(player2_id)}&APIkey={API_KEY}"
    response_1 = requests.get(complete_url_1)
    response_2 = requests.get(complete_url_2)
    player_1_data = response_1.json()
    player_2_data = response_2.json()

    player_1_df = pd.DataFrame(player_1_data['result'][0]['stats'])
    player_1_df = player_1_df[player_1_df['type'] == 'singles']
    player_1_df = player_1_df.replace('', 0)
    player_1_df[['rank', 'titles']] = player_1_df[['rank', 'titles']].astype(int)

    player_2_df = pd.DataFrame(player_2_data['result'][0]['stats'])
    player_2_df = player_2_df[player_2_df['type'] == 'singles']
    player_2_df = player_2_df.replace('', 0)
    player_2_df[['rank', 'titles']] = player_2_df[['rank', 'titles']].astype(int)

    df1 = player_1_df[['season', 'rank', 'titles']].iloc[::-1]
    df2 = player_2_df[['season', 'rank', 'titles']].iloc[::-1]
    df3 = df1.merge(df2, on='season', how='outer')
    df3.rename(columns={'titles_x': 'Player1_titles',
                        'titles_y': 'Player2_titles'},
               inplace=True, errors='raise')
    df3.replace(np.nan, 0, inplace=True)
    df3.sort_values(by=["season"], inplace=True)

    fig = px.bar(df3, x='season',
                 y=['Player1_titles', 'Player2_titles'], barmode="group")
    # Title
    annotations = [dict(xref='paper', yref='paper', x=0.0, y=1.05,
                        xanchor='left', yanchor='bottom',
                        text=f"Player 1: {player_1_data['result'][0]['player_name']} vs Player 2: "
                             f"{player_2_data['result'][0]['player_name']}  Seasonal Titles",
                        font=dict(family='Arial',
                                  size=30,
                                  color='rgb(37,37,37)'),
                        showarrow=False), dict(xref='paper', yref='paper', x=0.5, y=-0.25,
                                               xanchor='center', yanchor='top',
                                               text='Source: https://api-tennis.com/',
                                               font=dict(family='Arial',
                                                         size=12,
                                                         color='rgb(150,150,150)'),
                                               showarrow=False)]
    fig.update_layout(annotations=annotations)

    # fig.update_layout(template="plotly_white")
    return fig

def age(df, id_1, id_2):
    df['age'] = (pd.to_datetime("today") - pd.to_datetime(df["dob"])) / np.timedelta64(1, 'Y')
    age1 = df.loc[id_1]['age']
    age2 = df.loc[id_2]['age']
    ln1 = df.loc[id_1]['last_name']
    ln2 = df.loc[id_2]['last_name']
    
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "number",
        value = np.floor(age1),
        number = {'prefix': f"{ln1}: "},
        delta = {'position': "top", 'reference': 320},
        domain = {'x': [0, 1], 'y': [0.5, 1]}))

    fig.add_trace(go.Indicator(
        mode = "number",
        value = np.floor(age2),
        number = {'prefix': f"{ln2}: "},
        delta = {'position': "top", 'reference': 320},
        domain = {'x': [0, 1], 'y': [0, 1]}))

    return fig

def rank(df, standings, id_1, id_2):
    ln1 = df.loc[id_1]['last_name']
    ln2 = df.loc[id_2]['last_name']
    api_id1 = api_id(id_1)
    api_id2 = api_id(id_2)
    rank1 = int(standings[standings['player_key'] == str(api_id1)]["place"].to_list()[0])
    rank2 = int(standings[standings['player_key'] == str(api_id2)]["place"].to_list()[0])

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode = "number",
        value = np.floor(rank1),
        number = {'prefix': f"{ln1}: "},
        delta = {'position': "top", 'reference': 320},
        domain = {'x': [0, 1], 'y': [0.5, 1]}))

    fig.add_trace(go.Indicator(
        mode = "number",
        value = np.floor(rank2),
        number = {'prefix': f"{ln2}: "},
        delta = {'position': "top", 'reference': 320},
        domain = {'x': [0, 1], 'y': [0, 1]}))

    return fig

def summ_table(df, standings, id_1, id_2):
    df['age'] = (pd.to_datetime("today") - pd.to_datetime(df["dob"])) / np.timedelta64(1, 'Y')
    age1 = df.loc[id_1]['age']
    age2 = df.loc[id_2]['age']
    ln1 = df.loc[id_1]['last_name']
    ln2 = df.loc[id_2]['last_name']

    api_id1 = api_id(id_1)
    api_id2 = api_id(id_2)
    rank1 = int(standings[standings['player_key'] == str(api_id1)]["place"].to_list()[0])
    rank2 = int(standings[standings['player_key'] == str(api_id2)]["place"].to_list()[0])
    
    h1 = df.loc[id_1]['height']
    h2 = df.loc[id_2]['height'] 
    w1 = df.loc[id_1]['weight']
    w2 = df.loc[id_2]['weight']
    tp1 = df.loc[id_1]['turned_pro']
    tp2 = df.loc[id_2]['turned_pro']

    data = dict(values=[[int(age1), rank1, h1, w1, tp1], ['Age', "Rank", "Height", "Weight", "Turned Pro"], [int(age2), rank2, h2, w2, tp2]])

    fig = go.Figure(data=[go.Table(header=dict(values=[ln1, ' ', ln2]), cells=data)])

    return fig

def pred_winner(p1_id, p2_id, surface, bestof, indoor):
    winnerID = model.predict(p1_id,p2_id,surface,bestof,indoor)
    winnerLn = df.loc[winnerID]['last_name']

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "number",
        value = None,
        number = {'font_color':'white', 'font_size':10, 'prefix': f"Winner: {winnerLn}"},
        delta = {'position': "top", 'reference': 320},
        domain = {'x': [0, 1], 'y': [0.5, 1]}))
    fig.add_annotation(
        text=f"Winner: {winnerLn}",
        showarrow=False,
        font={'size': 60, 'color': 'black'})

    return fig

# ------------------------------------------------------ APP ------------------------------------------------------

app = dash.Dash(__name__, prevent_initial_callbacks=False)
server = app.server

# App Layout (HTML)

app.layout = html.Div(
    [
        # left pane
        html.Div(
            [   
                html.H1(children="Tennis Prediction"),
                html.Label(
                    "This app compares various statistics for two selected players and tries to predict the winner if "
                    "the selected players were to "
                    "face off, the dropdown option gives players who were active till the end of 2021 season. ",
                    style={"color": "rgb(33 36 35)"},
                ),
                html.Img(
                    src=app.get_asset_url("racquet.png"),
                    style={
                        "position": "relative",
                        "width": "160%",
                        "left": "-83px",
                        "top": "90px",
                    },
                ),
            ],
            className="side_bar",
        ),

        # player Selection Pane at Top
        html.Div(
            [
                html.Div(
                    [
                        html.Div([
                            html.Div([

                                html.Label("Select Player 1:"),
                                html.Br(),

                                html.Br(),
                                dcc.Dropdown(
                                    id='dropdown_player_1',
                                    options=[{'label': i, 'value': j} for i, j in dict(zip(df.name, df.index)).items()],
                                    value=4742),
                            ],

                                style={
                                    "margin": "10px",
                                    "display": "inline-block",
                                    "padding-top": "15px",
                                    "padding-bottom": "15px",
                                    "width": "12%",
                                }, ),

                            html.Img(
                                src=app.get_asset_url("player1.png"),
                                style={
                                    "position": "relative",
                                    "width": "5%",
                                    "left": "10px",
                                    "top": "20px",
                                    "display": "inline-block",
                                },
                            ),

                            html.Img(
                                id="player1_country_flag",
                                style={
                                    "position": "relative",
                                    "width": "10%",
                                    "left": "30px",
                                    "top": "20px",
                                    "display": "inline-block",
                                }),
                            html.Img(
                                id="player1_image",
                                style={
                                    "position": "relative",
                                    "width": "8%",
                                    "height": "auto",
                                    "left": "75px",
                                    "top": "20px",
                                    "display": 'inline-block'
                                }),

                            html.Img(
                                src=app.get_asset_url("vs.png"),
                                style={
                                    "position": "relative",
                                    "width": "5%",
                                    "left": "90px",
                                    "top": "20px",
                                    "display": "inline-block",
                                },
                            ),

                            html.Img(
                                id="player2_image",
                                style={
                                    "position": "relative",
                                    "width": "8%",
                                    "height": "auto",
                                    "left": "120px",
                                    "top": "20px",
                                    "display": 'inline-block'
                                }),

                            html.Img(
                                id="player2_country_flag",
                                style={
                                    "position": "relative",
                                    "width": "10%",
                                    "left": "170px",
                                    "top": "20px",
                                    "display": "inline-block",
                                }),

                            html.Img(
                                src=app.get_asset_url("player2.png"),
                                style={
                                    "position": "relative",
                                    "width": "5%",
                                    "left": "180px",
                                    "top": "30px",
                                    "display": "inline-block",
                                },
                            ),

                            html.Div([
                                html.Label("Select Player 2:"),
                                html.Br(),
                                html.Br(),
                                dcc.Dropdown(
                                    id='dropdown_player_2',
                                    options=[{'label': i, 'value': j} for i, j in dict(zip(df.name, df.index)).items()],
                                    value=6387),
                            ],

                                style={
                                    "margin": "10px",
                                    "display": "inline-block",
                                    "padding-top": "15px",
                                    "padding-bottom": "15px",
                                    "width": "12%",
                                    "position": "relative",
                                    "left": "200px",
                                }, ),

                        ], className="box"),
                        
                        # -----------------------Select parameters for prediction

                        html.Div([
                            html.Div([

                                html.Label("Select Surface:"),
                                html.Br(),
                                html.Br(),
                                dcc.RadioItems(id='surface_type',
                                               options=[
                                                {'label':'Grass', 'value':'G'},
                                                {'label':'Clay', 'value':'C'},
                                                {'label':'Hard', 'value':'H'}],
                                               value='G'
                                               )
                            ],
                                style={
                                    "margin": "5px",
                                    "display": "inline-block",
                                    "padding-top": "5px",
                                    "padding-bottom": "5px",
                                    "width": "25%",
                                }, ),

                            html.Div([
                                html.Label("Indoor or Outdoor:"),
                                html.Br(),
                                html.Br(),
                                dcc.RadioItems(id='in_out',
                                               options=[
                                                {'label':'Indoor', 'value': 1},
                                                {'label':'Outdoor', 'value':0}],
                                               value=1
                                               )
                            ],
                                style={
                                    "margin": "5px",
                                    "display": "inline-block",
                                    "padding-top": "5px",
                                    "padding-bottom": "5px",
                                    "width": "25%",
                                }, ),

                            html.Div([
                                html.Label("Best Of:"),
                                html.Br(),
                                html.Br(),
                                dcc.RadioItems(id='best_of',
                                               options=[
                                                   {'label':'Best of 3', 'value':3},
                                                   {'label':'Best of 5', 'value':5}
                                               ],
                                               value=3
                                               )
                            ],
                                style={
                                    "margin": "5px",
                                    "display": "inline-block",
                                    "padding-top": "5px",
                                    "padding-bottom": "5px",
                                    "width": "25%",
                                }, ),

                        ], className="box"),

                        # -----------------
                        #Row 5, Predicted Winner

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(id="title_bar8"),
                                                dcc.Graph(id="pred_winner"),
                                                html.Div(
                                                    [html.P(id="comment8", children='This indicator shows the predicted winner '
                                                                                    'using our ML model')],
                                                    className="box_comment",
                                                ),
                                            ],
                                            className="box",
                                            style={"padding-bottom": "15px"},
                                        ),

                                    ],
                                    style={"width": "100%", "display": "inline-block"},
                                ),

                            ],
                            className="box",
                        ),

                        # Two charts radar and map. Row 1 of viz

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(id="title_bar"),
                                                dcc.Graph(id="radar_chart"),
                                                html.Div(
                                                    [html.P(id="comment", children='Radar Chart comparing '
                                                                                   'two selected players on '
                                                                                   'performance metric of ratio of '
                                                                                   'matches won in different '
                                                                                   'tournaments and surfaces, '
                                                                                   'each color represent one player.')],
                                                    className="box_comment",
                                                ),
                                            ],
                                            className="box",
                                            style={"padding-bottom": "15px"},
                                        ),

                                    ],
                                    style={"width": "40%", "display": "inline-block"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(id="title_bar2"),
                                                dcc.Graph(id="map_chart"),
                                                html.Div(
                                                    [html.P(id="comment2", children="Choropleth map showing "
                                                                                    "the location of countries where "
                                                                                    "the two selected players are "
                                                                                    "from.")],
                                                    className="box_comment",
                                                ),

                                            ],
                                            className="box",
                                            style={"padding-bottom": "15px"},
                                        ),

                                    ],
                                    style={"width": "60%", "display": "inline-block"},
                                ),
                            ],
                            className="box",
                        ),

                        # Row 2 of viz

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(id="title_bar3"),
                                                dcc.Graph(id="rank_season"),
                                                html.Div(
                                                    [html.P(id="comment3", children='Line graphs shows rank of '
                                                                                    'player 1 and player 2 over '
                                                                                    'seasons')],
                                                    className="box_comment",
                                                ),
                                            ],
                                            className="box",
                                            style={"padding-bottom": "15px"},
                                        ),

                                    ],
                                    style={"width": "100%", "display": "inline-block"},
                                ),

                            ],
                            className="box",
                        ),

                        # Row 3 of viz

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(id="title_bar4"),
                                                dcc.Graph(id="title_season"),
                                                html.Div(
                                                    [html.P(id="comment4", children='Line graphs shows no of '
                                                                                    'titles won by '
                                                                                    'player 1 and player 2 over '
                                                                                    'seasons.')],
                                                    className="box_comment",
                                                ),
                                            ],
                                            className="box",
                                            style={"padding-bottom": "15px"},
                                        ),

                                    ],
                                    style={"width": "100%", "display": "inline-block"},
                                ),

                            ],
                            className="box",
                        ),

                        #Row 4, table with general info

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(id="title_bar7"),
                                                dcc.Graph(id="summ_table"),
                                                html.Div(
                                                    [html.P(id="comment7", children='This table shows general info about the '
                                                                                    'players')],
                                                    className="box_comment",
                                                ),
                                            ],
                                            className="box",
                                            style={"padding-bottom": "15px"},
                                        ),

                                    ],
                                    style={"width": "100%", "display": "inline-block"},
                                ),

                            ],
                            className="box",
                        ),
                        

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P(
                                            [
                                                "Team 180",
                                                html.Br(),
                                                "Sheikh Jalaluddin, Michael Rivera, Abanoub Abdelmalek, Mohammed Adel",
                                            ],
                                            style={"font-size": "12px"},
                                        ),
                                    ],
                                    style={"width": "60%"},
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            [
                                                "Data Source ",
                                                html.Br(),
                                                html.A(
                                                    "Ultimate Tennis Statistics",
                                                    href="https://hub.docker.com/r/mcekovic/uts-database",
                                                    target="_blank",
                                                ),
                                                ", ",
                                                html.A(
                                                    # "Second Reference",
                                                    # href="http://",
                                                    # target="_blank",
                                                ),
                                            ],
                                            style={"font-size": "12px"},
                                        )
                                    ],
                                    style={"width": "37%"},
                                ),
                            ],
                            className="footer",
                            style={"display": "flex"},
                        ),
                    ],
                    className="main",
                ),
            ]
        ),
    ]
)


# --------------------------- Callbacks ------------------------------------------------------

@app.callback(
    Output(component_id='radar_chart', component_property='figure'),
    [Input(component_id='dropdown_player_1', component_property='value'),
     Input(component_id='dropdown_player_2', component_property='value')])
def update_plot(player1, player2):
    if player1 != player2:
        fig = radar_chart(df_radar, player1, player2)
        fig.update_layout(template='gridon')
        return fig
    else:
        raise PreventUpdate


@app.callback(
    Output(component_id='map_chart', component_property='figure'),
    [Input(component_id='dropdown_player_1', component_property='value'),
     Input(component_id='dropdown_player_2', component_property='value')])
def update_plot(player1, player2):
    if player1 != player2:
        fig = show_country(player1, player2)
        fig.update_layout(template='gridon')
        return fig
    else:
        raise PreventUpdate


@app.callback(Output("player1_country_flag", "src"),
              [Input(component_id='dropdown_player_1', component_property='value')])
def update_flag2(player_id):
    if player_id is not None:
        return get_country_flag(player_id)
    else:
        raise PreventUpdate


@app.callback(Output("player2_country_flag", "src"),
              [Input(component_id='dropdown_player_2', component_property='value')])
def update_flag2(player_id):
    if player_id is not None:
        return get_country_flag(player_id)
    else:
        raise PreventUpdate


@app.callback(Output("player1_image", "src"),
              [Input(component_id='dropdown_player_1', component_property='value')])
def update_image(player_id):
    if player_id is not None:
        id_api = api_id(player_id)
        method_api = 'method=get_players'
        full_url = f"{BASE_URL}{method_api}&player_key={id_api}&APIkey={API_KEY}"
        response = requests.get(full_url)
        player_data = response.json()
        image_url = player_data['result'][0]['player_logo']
        return image_url
    else:
        raise PreventUpdate


@app.callback(Output("player2_image", "src"),
              [Input(component_id='dropdown_player_2', component_property='value')])
def update_image(player_id):
    if player_id is not None:
        id_api = api_id(player_id)
        method_api = 'method=get_players'
        full_url = f"{BASE_URL}{method_api}&player_key={id_api}&APIkey={API_KEY}"
        response = requests.get(full_url)
        player_data = response.json()
        image_url = player_data['result'][0]['player_logo']
        return image_url
    else:
        raise PreventUpdate


@app.callback(
    Output(component_id='rank_season', component_property='figure'),
    [Input(component_id='dropdown_player_1', component_property='value'),
     Input(component_id='dropdown_player_2', component_property='value')])
def update_plot(player1, player2):
    if player1 != player2:
        fig = rank_season(player1, player2)
        fig.update_layout(template='gridon')
        return fig
    else:
        raise PreventUpdate


@app.callback(
    Output(component_id='title_season', component_property='figure'),
    [Input(component_id='dropdown_player_1', component_property='value'),
     Input(component_id='dropdown_player_2', component_property='value')])
def update_plot(player1, player2):
    if player1 != player2:
        fig = title_season(player1, player2)
        fig.update_layout(template='gridon')
        return fig
    else:
        raise PreventUpdate

@app.callback(
    Output(component_id='summ_table', component_property='figure'),
    [Input(component_id='dropdown_player_1', component_property='value'),
     Input(component_id='dropdown_player_2', component_property='value')])
def update_plot(player1, player2):
    if player1 != player2:
        fig = summ_table(df, standings, player1, player2)
        fig.update_layout(template='gridon')
        return fig
    else:
        raise PreventUpdate

@app.callback(
    Output(component_id='pred_winner', component_property='figure'),
    [Input(component_id='dropdown_player_1', component_property='value'),
     Input(component_id='dropdown_player_2', component_property='value'),
     Input(component_id='surface_type', component_property='value'),
     Input(component_id='best_of', component_property='value'),
     Input(component_id='in_out', component_property='value')
     ])
def update_plot(player1, player2, surface_type, best_of, in_out):
    if player1 != player2:
        fig = pred_winner(player1, player2, surface_type, best_of, in_out)
        fig.update_layout(template='gridon')
        return fig
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)
