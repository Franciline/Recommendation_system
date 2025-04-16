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

df = pd.DataFrame(data={
    "x": games_tsne[:, 0].tolist(),
    "y": games_tsne[:, 1].tolist(),
    "z": games_tsne[:, 2].tolist(),

    "color": colors_points,
    "cluster": clusters,
})

# View
target = [df.x.mean(), df.y.mean(), df.z.mean()]
initial_view_state = pdk.ViewState(
    latitude=0, longitude=0, target=target, controller=True, rotation_x=15, rotation_orbit=30, zoom=0)

# TSNE using PyDeck
arc_layer = pdk.Layer(
    "PointCloudLayer",
    data=df,
    get_position=["x", "y", "z"],
    get_color="color",
    pickable=True,
    auto_highlight=True,
    point_size=2
)

deck = pdk.Deck(
    layers=[arc_layer],
    initial_view_state=initial_view_state,
    views=[pdk.View(type="OrbitView", controller=True)],
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
                 mapboxKey=mapbox_key
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
            ], style={"height": "20%", "padding": "20px"})


        ],
            style={
                "width": "50vh",
                # "borderLeft": "1px solid #ddd",
                "backgroundColor": "#7d808f",
                # "background": "linear-gradient(270deg, #7d808f, 75%, #ffffff)",
                "height": "100vh",
                "overflowY": "auto"
        },
            className="sidebar"
        ),

        html.Div(className="gradient-edge")

    ], style={"display": "flex", "flexDirection": "row"})
])


# Select thematic clusters based on dropdown
@app.callback(
    Output("tsne", "data"),
    Input("themes-dropdown", "value"),
    prevent_initial_call=True,
)
def update_plot(value):
    if value is None:
        return

    if value == "Tout":
        points = df
        view_state = initial_view_state

    else:
        points = df[df["cluster"].isin(themes[value])]
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


if __name__ == '__main__':
    app.run(debug=True)
