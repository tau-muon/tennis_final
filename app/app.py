# Import Libraries

import os

import dash
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash import html, dcc
from dash.dependencies import Input, Output

# Read the data

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "data/")

df = pd.read_csv(path + "tennis_players_data.csv", index_col="player_id")
df = df[df.active == "t"]
df_radar = df[["name", "matches_win_percentage", "grand_slam_win_percentage", "tour_finals_win_percentage",
               "olympics_matches_win_percentage", "davis_cup_matches_win_percentage", "hard_matches_win_percentage",
               "clay_matches_win_percentage",
               "grass_matches_win_percentage", "carpet_matches_win_percentage", "outdoor_matches_win_percentage",
               "indoor_matches_win_percentage"]]

country_code = dict(df["country_id"])


# Function for figures

def radar_chart(data, player_id_1, player_id_2):
    df_graph = data[data.index.isin([player_id_1, player_id_2])]

    categories = ['Matches Won', 'Grand Slam Matches Won', 'Tour Finals Matches Won',
                  'Olympics Matches Won', 'Davius Cup Matches Won', 'Hard Matches Won',
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
    fig.update_layout(height=450, width=600)
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
                    "This app compares two players and tries to predict the winner if the selected players were to "
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
                                    value=3819),
                            ],

                                style={
                                    "margin": "10px",
                                    "display": "inline-block",
                                    "padding-top": "15px",
                                    "padding-bottom": "15px",
                                    "width": "30%",
                                }, ),
                            html.Img(
                                src=app.get_asset_url("player1.png"),
                                style={
                                    "position": "relative",
                                    "width": "5%",
                                    "left": "10px",
                                    "top": "50px",
                                },
                            ),
                            html.Div([
                                html.Label("Select Player 2:"),
                                html.Br(),
                                html.Br(),
                                dcc.Dropdown(
                                    id='dropdown_player_2',
                                    options=[{'label': i, 'value': j} for i, j in dict(zip(df.name, df.index)).items()],
                                    value=3333),
                            ],

                                style={
                                    "margin": "10px",
                                    "display": "inline-block",
                                    "padding-top": "15px",
                                    "padding-bottom": "15px",
                                    "width": "30%",
                                    "position": "relative",
                                    "left": "150px",
                                }, ),
                            html.Img(
                                src=app.get_asset_url("player2.png"),
                                style={
                                    "position": "relative",
                                    "width": "5%",
                                    "left": "200px",
                                    "top": "50px",
                                },
                            ),

                        ], className="box"),

                        # Two charts radar and map

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(id="title_bar"),
                                                dcc.Graph(id="radar_chart"),
                                                html.Div(
                                                    [html.P(id="comment")],
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
                                                    [html.P(id="comment2")],
                                                    className="box_comment",
                                                ),

                                            ],
                                            className="box",
                                            style={"padding-bottom": "15px"},
                                        ),

                                    ],
                                    style={"width": "40%", "display": "inline-block"},
                                ),
                            ],
                            className="box",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Empty Box",
                                            style={"font-size": "medium"},
                                        ),
                                        html.Br(),

                                        html.Br(),

                                    ],
                                    className="box",
                                    style={"width": "40%"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Empty Box",
                                            style={"font-size": "medium"},
                                        ),
                                        html.Br(),
                                        html.Br(),

                                    ],
                                    className="box",
                                    style={"width": "63%"},
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P(
                                            [
                                                "Team 180",
                                                html.Br(),
                                                "Sheikh Jalaluddin, Michael Riveira, Abanoub Abdelmalek, Mohammed Adel",
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
                                                    # "Second Refreence",
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
    fig = radar_chart(df_radar, player1, player2)
    fig.update_layout(template='gridon')
    return fig



@app.callback(
    Output(component_id='map_chart', component_property='figure'),
    [Input(component_id='dropdown_player_1',component_property= 'value'),
    Input(component_id='dropdown_player_2',component_property= 'value')])
def update_plot(player1, player2):
    fig = show_country(player1, player2)
    fig.update_layout(template='gridon')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
