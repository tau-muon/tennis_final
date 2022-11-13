#import libraries

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd


# Read data

df = pd.read_csv("tennis_players_data.csv", index_col="player_id")
df = df[df.active == "t"]
df_radar = df[["name","matches_win_percentage", "grand_slam_win_percentage","tour_finals_win_percentage",\
                                                  "olympics_matches_win_percentage","davis_cup_matches_win_percentage", "hard_matches_win_percentage", "clay_matches_win_percentage",\
                                                  "grass_matches_win_percentage","carpet_matches_win_percentage", "outdoor_matches_win_percentage", "indoor_matches_win_percentage" ]]

# Crate app

app = dash.Dash(__name__, prevent_initial_callbacks=False)
server = app.server

#define graph function

def radar_chart(data, player_id_1, player_id_2):
    
    df_graph = data[data.index.isin([player_id_1,player_id_2])]
    
    categories = ['Matches Won','Grand Slam Matches Won','Tour Finals Matches Won',
              'Olympics Matches Won','Davius Cup Matches Won','Hard Matches Won',
              'Clay Matches Won','Grass Matches Won','Carper Matches Won',
              'Outdoor Matches Won','Indoor Matches Won' ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
          r=list(df_graph.iloc[0, 1:] ),
          theta=categories,
          fill='toself',
          name= df_graph.iloc[0, 0]
    ))
    fig.add_trace(go.Scatterpolar(
          r = list(df_graph.iloc[1, 1:] ),
          theta=categories,
          fill='toself',
          name= df_graph.iloc[1, 0]
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=True,

    )

    fig.update_layout(title_text='Radar Chart', title_x=0.5,title_y= 0.95, title_font_family= "Old Standard TT",title_font_size= 40 , title_font_color= 'green', paper_bgcolor= 'white', plot_bgcolor = 'red' )
    
    return fig
    
    
# App layout

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='"Tennis Visualization App"',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: Visual comparision of tennis playets', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
    html.Label("Select Player_1"),
    dcc.Dropdown(
               id='dropdown_player_1',
               options=[{'label': i, 'value': j} for i,j in dict( zip(df.name , df.index)).items()],
               value = 3819),
    
    html.Label("Select Player_2"),
    dcc.Dropdown(
               id='dropdown_player_2',
               options=[{'label': i, 'value': j} for i,j in dict( zip(df.name , df.index)).items()],
               value = 3333),

    dcc.Graph(id='radar_chart', style={'height': '70vh'} )

    #dash_table.DataTable( df_radar )
        

])

#app call back


@app.callback(
    Output(component_id='radar_chart', component_property='figure'),
    [Input(component_id='dropdown_player_1',component_property= 'value'),
    Input(component_id='dropdown_player_2',component_property= 'value')])

def update_plot(player1, player2):
    
    fig = radar_chart(df_radar, player1, player2)
    
    return fig

# Run the app

if __name__ == '__main__':
    app.run_server(debug=True)
