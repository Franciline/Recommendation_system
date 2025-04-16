# Requires Dash 2.17.0 or later

"""
Clusters grouped together :

-------
Worst rated cluster
10
392f5a
-------

-------
Construction. Batiments. Jeunes/famille
0 - 25 - 12 - 19
ca5310 bb4d00 8f250c 691e06 -> Brown
-------

-------
Rapide. Ressources, tactique
20 - 27
c05299 822faf -> Purple
-------

-------
Collection (trésors, cartes)
17 - 14 - 1 - 8 - 4
ffea00 ffdd00 ffd000 ffc300 ffb700 ffaa00 -> Yellow
-------

-------
Orienté sur le réflexion (logique, puzzle, détective, réponse aux questions)
23 - 9 - 6 - 3 (y compris échecs) - 29 (~bluffing, role playing) - 26

90caf9 2196f3 1e88e5 1976d2 1565c0 0d47a1 -> Blue
-------

-------
Orienté sur le gestion des ressources, colonisation, guerre, différentes époques
13 (difficle à comprendre, présence des animaux, centroide est proche à 5)
5 - 13 - 18 - 28 (différentes époques) - 22 - 11 - 16 - 21 - 15

5 : 10451d
155d27 1a7431 208b3a 25a244 2dc653 4ad66d 6ede8a 92e6a7 b7efc5 -> Green
-------

-------
Jeux longs & complexes
2 - 7 (great visuals aussi)
60d394 aaf683 -> Green bright
-------

-------
Best rated
24
-------
"""
import os

import dash_bootstrap_components as dbc
from plotly.colors import qualitative
from dash import Dash, html, dcc, callback, Output, Input, State, ctx, ALL, MATCH
import pandas as pd
import numpy as np
import pydeck as pdk
import dash_deck
from PIL import ImageColor

mapbox_key = os.getenv("MAPBOX_ACCESS_TOKEN")

n_clusters = 30  # -> 30 colors to generate
offset = 200
stretch = [1., 1., 1.]
games_tsne = np.load("tsne_pushed.npy", mmap_mode="r")
clusters = np.load("clusters.npy", mmap_mode="r")
games_info = pd.read_csv("games_info.csv", index_col=0)

current_cluster = None  # TO ADD


# colors = np.array((qualitative.Plotly + qualitative.Set3 + qualitative.Alphabet)[:n_clusters])
colors = {0: "#ca5310", 25: "#bb4d00", 12: "#8f250c", 19: "#691e06",
          20: "#c05299", 27: "#822faf",
          17: "#ffea00", 14: "#ffdd00", 1: "#ffc300", 8: "#ffb700", 4: "#ffaa00",
          23: "#90caf9", 9: "#2196f3", 6: "#1e88e5", 3: "#1976d2", 29: "#1565c0", 26: "#0d47a1",
          5: "#10451d", 13: "#155d27", 18: "#1a7431", 28: "#208b3a", 22: "#25a244", 11: "#2dc653", 16: "#4ad66d", 21: "#6ede8a", 15: "#92e6a7",
          2: "#60d394", 7: "#aaf683",
          24: "#e53d00", 10: "#392f5a"
          }

# Coloring : hex to rgb
colors_points = [list(ImageColor.getcolor(colors[cluster], "RGB")) for cluster in clusters]

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# app = Dash

df_all = pd.DataFrame(data={
    "game_id": games_info["Game id"],

    "x": games_tsne[:, 0].tolist(),
    "y": games_tsne[:, 1].tolist(),
    "z": games_tsne[:, 2].tolist(),

    "color": colors_points,
    "cluster": clusters,
    "name": clusters,
})

# View
target = [df_all.x.mean(), df_all.y.mean(), df_all.z.mean()]
initial_view_state = pdk.ViewState(
    latitude=0, longitude=0, target=target, controller=True, rotation_x=15, rotation_orbit=30, zoom=0)

# TSNE using PyDeck
arc_layer = pdk.Layer(
    "PointCloudLayer",
    data=df_all,
    get_position=["x", "y", "z"],
    get_color="color",
    pickable=True,  # enable hover
    auto_highlight=True,
    point_size=2
)

# Text to display on hover
tooltip = {
    "html": "{cluster}"
}

deck = pdk.Deck(
    layers=[arc_layer],
    initial_view_state=initial_view_state,
    views=[pdk.View(type="OrbitView", controller=True)],
    tooltip=tooltip
)


themes = {
    "Tout": [],
    "🧩🕵️ Logic et Déduction": [23, 9, 6, 3, 29, 26],
    "🏗️👨‍👩‍👧‍👦 Construction": [0, 25, 12, 19],
    "🧠⚡ Rapide et Tactique": [20, 27],
    "📚⏳ Longs et complexes": [2, 7],
    "💎🃏 Collecte":  [17, 14, 1, 8, 4],
    "🏰🕰️ Historique": [5, 13, 18, 28, 22, 11, 16, 21, 15],
    "👎 Coups de blues": [10],
    "🏆 Coups de coeurs": [24],
}


app.layout = html.Div([

    html.Div([
        # TSNE on the left
        html.Div([  # Left: DeckGL
             dash_deck.DeckGL(
                 id="tsne",
                 data=deck.to_json(),
                 tooltip={"text": "{name}"},
                 # style={"flex": "1", "height": "100%"},
                 mapboxKey=mapbox_key,
                 enableEvents=["click", "hover"]
             )],
            style={"flex": "1", "minWidth": "0", "margin": "0 auto",
                   "overflow": "hidden", "position": 'relative', "width": "50vw"}
        ),

        # Right sidebar
        html.Div([
            html.Div([
                dcc.Dropdown([{"label": html.Span(t, style={}), "value": t} for t in themes],
                             'Tout', id='themes-dropdown', searchable=False, clearable=False, className="rounded-dropdown",
                             ),
            ], style={"height": "10%", "padding": "20px", "backgroundColor": "#f2e9e4}"}),

            html.Div(children=[], id="game-info-div", style={"display": "none"})

        ], style={
            "width": "60vh",
            # "borderLeft": "1px solid #ddd",
            "backgroundColor": "#22223b",
            # "background": "linear-gradient(270deg, #7d808f, 75%, #ffffff)",
            "height": "100vh",
            "overflowY": "auto"
        },

        ),


        html.Div(className="gradient-edge")

    ], style={"display": "flex", "flexDirection": "row"})
])


# Select thematic clusters based on dropdown
@app.callback(
    Output("tsne", "data", allow_duplicate=True),
    Input("themes-dropdown", "value"),
    prevent_initial_call=True,
)
def update_plot(value):
    global current_cluster, df_all
    if value is None:
        return
    specific_cluster_shown = False
    current_cluster = None

    if value == "Tout":
        points = df_all
        view_state = initial_view_state

    else:
        points = df_all[df_all["cluster"].isin(themes[value])]
        target = [points.x.mean(), points.y.mean(), points.z.mean()]
        view_state = pdk.ViewState(
            latitude=target[0], longitude=target[1], target=target, controller=True, rotation_x=0, rotation_orbit=0, zoom=1.5)

    layer = pdk.Layer(
        "PointCloudLayer",
        data=points,
        get_position=["x", "y", "z"],
        get_color="color",
        pickable=True,
        auto_highlight=True,
        point_size=2
    )

    new_deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        views=[pdk.View(type="OrbitView", controller=True)],
    )

    return new_deck.to_json()


# Click on a cluster -> zoom on this cluster, hide clusters points


@app.callback(
    Output('tsne', 'data', allow_duplicate=True),
    Input('tsne', 'clickInfo'),
    prevent_initial_call=True
)
def event_handler(clickInfo):

    global current_cluster
    # 1st condition : click on point in PointCloud. 2nd condition allows not to change cluster if only one cluster shown.

    # TO DO: combine these conditions
    if (clickInfo is None):
        return

    if ("object" not in clickInfo):
        return

    if clickInfo["object"] is None:
        return

    clicked_cluster = clickInfo["object"]["cluster"]
    if current_cluster == clicked_cluster:
        return

    current_cluster = clicked_cluster
    points = df_all[df_all["cluster"] == clicked_cluster]

    target = [points.x.mean(), points.y.mean(), points.z.mean()]
    view_state = pdk.ViewState(
        latitude=target[0], longitude=target[1], target=target, controller=True, rotation_x=0, rotation_orbit=0, zoom=4)

    layer = pdk.Layer(
        "PointCloudLayer",
        data=points,
        get_position=["x", "y", "z"],
        get_color="color",
        pickable=True,
        auto_highlight=True,
        point_size=4
    )

    new_deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        views=[pdk.View(type="OrbitView", controller=True)],
    )

    return new_deck.to_json()


# Click on a point when one cluster is observed
@app.callback(
    Output("game-info-div", "children"),
    Output("game-info-div", "style"),
    Input('tsne', 'clickInfo'),
    prevent_initial_call=True
)
def display_game_info(click_info):
    global current_cluster
    if click_info is None:
        return [], {"display": "none"},

    if click_info["object"] is None:
        return [], {"display": "none"}

    game_info = games_info[games_info["Game id"] == click_info["object"]["game_id"]].iloc[0].to_dict()
    print(game_info["Description"])
    return [html.H3(html.B(game_info["Game name year"]), style={"margin-bottom": "20px"}),
            html.Div([html.P([html.B("Rating: "), f"{game_info['Rating']:.2f}"]),
                      html.P([html.B(f"Type: "), game_info['Type']]),
                      html.P([html.B(f"Joueurs: "), game_info['Players']]),
                      html.P([html.B(f"Age: "), game_info['Age']])],
                     style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gridGap": "5px", "width": "100%"}),
            html.P(game_info["Description"], style={"textAlign": "justify"})], {"display": "block", "width": "100%", "color": "#e5e5e5", "padding": "25px"}


if __name__ == '__main__':
    app.run(debug=True)
