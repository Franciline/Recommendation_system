# Requires Dash 2.17.0 or later
"""
TO DO :
- Exploration mode    : print summary of comments on hover
- Recommendation mode : consider printing comment that user gave to a game)
"""

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

from PIL import ImageColor
import dash_deck
import pydeck as pdk
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State, ctx, ALL, MATCH, no_update
from plotly.colors import qualitative
import os
import dash_bootstrap_components as dbc


def get_deck(points, view, point_size=2.5):
    layer = pdk.Layer(
        "PointCloudLayer",
        data=points,
        get_position=["x", "y", "z"],
        get_color="color",
        pickable=True,  # enable hover
        auto_highlight=True,
        point_size=point_size,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        views=[pdk.View(type="OrbitView", controller=True)],
        # tooltip=tooltip  # Text to display on hover

    )
    return deck


def compute_point_size(n_points, min_points=150, max_points=600, min_size=2.5, max_size=4.5):
    # n_points to range between [min_points, max_points]
    n_clipped = max(min_points, min(n_points, max_points))

    # Normalize between 0 and 1 (inverted: more points = lower size)
    norm = (max_points - n_clipped) / (max_points - min_points)

    # Scale to point size range
    size = min_size + norm * (max_size - min_size)
    print(f"Point size for {n_points} points", size)
    return size


def compute_zoom(n_points, min_points=150, max_points=400, min_zoom_level=0, max_zoom_level=4):
    n_clipped = max(min_points, min(n_points, max_points))

    norm = (max_points - n_clipped) / (max_points - min_points)

    zoom = min_zoom_level + norm * (max_zoom_level - min_zoom_level)
    print(f"Estimated zoom on {n_points} points", zoom)
    return zoom


def get_view_params(points):
    """
    Get parameters for ViewState in order :
        - latitude
        - longitude
        - target
    """

    x_mean, y_mean, z_mean = points.x.mean(), points.y.mean(), points.z.mean()
    return x_mean, y_mean, [x_mean, y_mean, z_mean]


def recalc_view(points, game_info, initial_view_state):
    if points.shape[0] == games_info.shape[0]:  # All points
        view = initial_view_state
    else:
        lat, lon, target = get_view_params(points)
        view = pdk.ViewState(latitude=lat, longitude=lon, target=target,
                             controller=True, rotation_x=0, rotation_orbit=0, zoom=compute_zoom(points.shape[0]))
    return view

    # def view_state_to_dict(view_state):
    #     return {
    #         "latitude": view_state.latitude,
    #         "longitude": view_state.longitude,
    #         "target": view_state.target,
    #         "controller": view_state.controller,
    #         "rotation_x": view_state.rotation_x,
    #         "rotation_orbit": view_state.rotation_orbit,
    #         "zoom": view_state.zoom,
    #     }

    # def dict_to_view_state(state_dict):
    #     return pdk.ViewState(
    #         latitude=state_dict["latitude"],
    #         longitude=state_dict["longitude"],
    #         target=state_dict["target"],
    #         controller=state_dict["controller"],
    #         rotation_x=state_dict["rotation_x"],
    #         rotation_orbit=state_dict["rotation_orbit"],
    #         zoom=state_dict["zoom"]
    #     )


n_clusters = 30  # -> 30 colors to generate
games_tsne = np.load("tsne_pushed.npy", mmap_mode="r")   # TSNE 3D
clusters = np.load("clusters.npy", mmap_mode="r")        # Clusters assignment

# NNMF prediction U @ G.T (already existing ratings are replaced by true ones)
nmf_pred = np.load("nnmf_prediction.npy", mmap_mode="r")

games_info = pd.read_csv("games_info.csv", index_col=0)  # info on games
users_info = pd.read_parquet("users_info.parquet")  # info on users


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

# TO DO : once OK -> save to file
df_all = pd.DataFrame(data={
    "game_id": games_info["Game id"],

    "x": games_tsne[:, 0].tolist(),
    "y": games_tsne[:, 1].tolist(),
    "z": games_tsne[:, 2].tolist(),

    "color": colors_points,
    "cluster": clusters,
    "name": clusters,
    "game index": games_info["Game index"]
})

# View
lat, lon, target = get_view_params(df_all)
initial_view_state = pdk.ViewState(latitude=lat, longitude=lon, target=target,
                                   controller=True, rotation_x=15, rotation_orbit=30, zoom=0)

deck = get_deck(df_all, initial_view_state)

# For themes-dropdown
themes = {
    "Tout les clusters": [],
    "🧩🕵️ Logique et Déduction": [23, 9, 6, 3, 29, 26],
    "🏗️👨‍👩‍👧‍👦 Construction": [0, 25, 12, 19],
    "🧠⚡ Rapide et Tactique": [20, 27],
    "📚⏳ Longs et complexes": [2, 7],
    "💎🃏 Collecte":  [17, 14, 1, 8, 4],
    "🏰🕰️ Historique": [5, 13, 18, 28, 22, 11, 16, 21, 15],
    "👎 Coups de blues": [10],
    "🏆 Coups de coeurs": [24],
}
themes_inverse = {cluster: theme for theme, clusters in themes.items() for cluster in clusters}

# Main layout : TSNE (DeckGL) on the left, sidebar on the right
app.layout = html.Div([
    dcc.Store(id="explore-mode", data=True),  # True if exploration mode, False if recommendation mode
    dcc.Store(id="plotted-data", data=df_all["game index"].values),  # store plotted games indices
    dcc.Store(id="current-cluster", data=None),  # allows not to change cluster if only one cluster shown.

    # makes sense in reco mode. Index of current chosen user. Necessary to show recommended games
    dcc.Store(id="current-user", data=None),

    # interval to allow animation of closing dropdown-reco-games
    dcc.Interval(id='clear-dropdown-interval', interval=400, n_intervals=0, disabled=True),

    html.Div([
        # Header when only one cluster is shown
        html.Div(children="Tout les clusters", id="cluster-theme-header", className='cluster-theme-header'),

        html.Div([
             dash_deck.DeckGL(
                 id="tsne",
                 data=deck.to_json(),
                 tooltip={"text": "{name}"},
                 # style={"flex": "1", "height": "100%"},
                 enableEvents=["click", "hover"]
             )],
            style={"flex": "1", "minWidth": "0", "margin": "0 auto",
                   "overflow": "hidden", "position": 'relative', "width": "50vw"}
        ),

        # Right sidebar
        html.Div([
            html.Div([
                html.Button("🔁 Recommandation", id="mode-button"),  # button to change mode : reco ou exploration
                dcc.Dropdown(options=[{"label": t, "value": t} for t in themes],
                             value='Tout les clusters', id='themes-dropdown', searchable=False, clearable=False, className="rounded-dropdown",
                             style={"width": "100%", "flex": "1"})
            ], style={"padding": "20px", "display": "flex", "width": "100%", 'alignItems': 'center', "height": "fitContent"}),

            html.Div(children=[dcc.Dropdown(options=[{"label": "Choisir un profile", "value": "..."}] +
                                            [{"label": f"User {username}", "value": index}
                                             for username, index in users_info[["Username", "User index"]].itertuples(index=False)],
                                            value="...", searchable=True, clearable=False, id="users-dropdown", disabled=True,
                                            style={"width": "95%", "flex": "1"}),
                               html.Div([html.Button('←', id="button-reco-games", className="button-reco-games", disabled=True)])],
                     id="user-select-div",
                     style={"height": "auto", "padding": "0px 20px 10px 20px", "display": "flex"}),
            html.Div(id="dropdown-reco-games", children=[], className="dropdown-reco-games"),
            html.Div(children=[], id="game-info-div", style={"display": "none"})

        ], style={
            "width": "60vh",
            "backgroundColor": "#22223b",
            "height": "100vh",
            "overflowY": "auto",
            "maxWidth": "60vh",
        },
        ),
    ], style={"display": "flex", "flexDirection": "row", "overflowX": "auto", "minWidth": "800px"})
])


# Select thematic clusters based on dropdown
@app.callback(
    Output('tsne', 'data', allow_duplicate=True),
    Output('plotted-data', 'data', allow_duplicate=True),
    Output('current-cluster', 'data', allow_duplicate=True),
    Output('cluster-theme-header', 'children', allow_duplicate=True),

    Input('themes-dropdown', 'value'),
    prevent_initial_call=True,
)
def thematic_clusters(value):
    """Plot only selected thematic clusters."""
    global df_all

    if value is None:
        return no_update, no_update, no_update, no_update

    if value == "Tout les clusters":
        points = df_all
        view_state = initial_view_state

    else:
        points = df_all[df_all["cluster"].isin(themes[value])]
        target = [points.x.mean(), points.y.mean(), points.z.mean()]
        lat, lon, target = get_view_params(points)

        view_state = pdk.ViewState(latitude=lat, longitude=lon, target=target,
                                   controller=True, rotation_x=0, rotation_orbit=0, zoom=1.5)

    new_deck = get_deck(points, view_state, compute_point_size(points.shape[0]))
    return new_deck.to_json(), points["game index"].values, None, value


# Click on a cluster -> zoom on this cluster, hide clusters points


@app.callback(
    Output('themes-dropdown', 'value'),
    Output('tsne', 'data', allow_duplicate=True),
    Output("plotted-data", "data", allow_duplicate=True),
    Output('current-cluster', 'data', allow_duplicate=True),
    Output('cluster-theme-header', 'children', allow_duplicate=True),

    Input('tsne', 'clickInfo'),
    State('current-cluster', 'data'),
    prevent_initial_call=True
)
def zoom_cluster(clickInfo, current_cluster):
    """Plot only one cluster on a click on one of its points"""
    if (clickInfo is None) or ("object" not in clickInfo) or clickInfo["object"] is None:
        return no_update, no_update, no_update, no_update, no_update

    clicked_cluster = clickInfo["object"]["cluster"]
    if current_cluster == clicked_cluster:
        return no_update, no_update, no_update, no_update, no_update

    points = df_all[df_all["cluster"] == clicked_cluster]

    lat, lon, target = get_view_params(points)
    view_state = pdk.ViewState(latitude=lat, longitude=lon, target=target,
                               controller=True, rotation_x=0, rotation_orbit=0, zoom=4)

    new_deck = get_deck(points, view_state, compute_point_size(points.shape[0]))
    return None, new_deck.to_json(), points["game index"].values, clicked_cluster, themes_inverse[clicked_cluster]


# Click on a point when one cluster is observed
@app.callback(
    Output("game-info-div", "children"),
    Output("game-info-div", "style"),
    Input('tsne', 'clickInfo'),
    prevent_initial_call=True
)
def display_game_info(click_info):
    """Add selected game summary on a sidebar"""
    global current_cluster
    if click_info is None:
        return [], {"display": "none"},

    if click_info["object"] is None:
        return [], {"display": "none"}

    game_info = games_info[games_info["Game id"] == click_info["object"]["game_id"]].iloc[0].to_dict()
    return [html.Div([html.H4(html.B(game_info["Game name year"]), style={"margin-bottom": "20px", "width": "80%"}),
                      html.H3([f"{game_info['Rating']:.1f}", " ⭐"], style={"width": "20%", "textAlign": "right"})],
                     style={"display": "flex", "justifyContent": "spaceBetween"}),
            html.P(html.P([html.B("🎲 Type: "), game_info['Type']])),
            html.Div([html.P([html.B("👨‍👩‍👧‍👦 Joueurs: "), game_info['Players']]),
                      html.P([html.B("🎂 Age: "), game_info['Age']])],


                     style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "width": "100%"}),
            html.P("TO DO: Comment summary here", style={"textAlign": "justify"})], {
                "display": "block",
                "width": "auto",
                "color": "#14213d",
                "padding": "25px",
                "backgroundColor": "#ffffff",
                "padding": "20px",
                "margin-left": "20px",
                "margin-right": "20px",
                "borderRadius": "15px",

    }


@app.callback(
    Output('users-dropdown', 'disabled'),
    Output('users-dropdown', 'value'),
    Output('mode-button', 'children'),
    Output('explore-mode', 'data'),
    Output('tsne', 'data'),
    Output('dropdown-reco-games', 'className', allow_duplicate=True),
    Output('button-reco-games', 'className', allow_duplicate=True),

    Input('mode-button', 'n_clicks'),  # n_clicks is not used, but Dash demands non empty property
    State('explore-mode', 'data'),
    State('plotted-data', 'data'),
    prevent_initial_call=True
)
def change_mode(click_info, mode_data, plotted_games_index):
    """Change mode :
        1. recommendation -> exploration or
        2. exploration    -> recommendation

    Change of colors on df is applied only if 'recommendation -> exploration'
    """

    global themes, df_all, colors_points, initial_view_state
    if click_info is None:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    print(len(plotted_games_index))
    points = df_all[df_all["game index"].isin(plotted_games_index)]

    # Exploration mode -> go to Reco mode
    if mode_data == True:
        points["name"] = ""  # nothing on hover
        return False, "...", "🔁 Exploration", False, no_update, no_update, no_update

    # Reco mode -> go to Exploration mode. Dropdown : cluster themes
    df_all["color"] = colors_points  # back to palette color
    view = recalc_view(points, games_info, initial_view_state)
    new_deck = get_deck(points, view, compute_point_size(points.shape[0]))

    return True, "...", "🔁 Recommandation", True, new_deck.to_json(), 'dropdown-reco-games', 'button-reco-games'


@app.callback(
    Output('tsne', 'data', allow_duplicate=True),
    Output('tsne', 'tooltip'),
    Output('current-user', 'data'),
    Output('button-reco-games', 'disabled'),
    Output('dropdown-reco-games', 'className'),
    Output('button-reco-games', 'className'),

    Input("users-dropdown", "value"),
    State("plotted-data", "data"),
    prevent_initial_call=True
)
def get_user_tsne(user_index, plotted_games_index):
    """Replot plotted games to change their color based on predicted ratings with NNMF

    Note :
        !All! colors in 'df_all' are changed since it is more logical for a user to go explore other points
    """

    global initial_view_state, df_all

    if user_index is None or plotted_games_index is None:
        return no_update, no_update, no_update, no_update, no_update

    points = df_all[df_all["game index"].isin(plotted_games_index)]
    view = recalc_view(points, games_info, initial_view_state)

    if user_index == "...":  # No user selected
        points["name"] = ""
        df_all["color"] = colors_points

        new_deck = get_deck(points, view, compute_point_size(points.shape[0]))
        return new_deck.to_json(), {"text": "{name}"}, no_update, True, 'dropdown-reco-games', 'button-reco-games'

    # No user change -> the event is not fired

    ratings = nmf_pred[user_index, :]

    # RGB. Rating 0 = red, Rating 10 = green
    red = (255 * (1 - ratings)).astype(int)
    green = (255 * ratings).astype(int)
    blue = np.zeros_like(red)

    # Reassign colors
    colors = np.stack([red, green, blue], axis=1).tolist()
    df_all["color"] = colors

    # Color rated games to blue
    user_info = users_info[users_info["User index"] == user_index]

    mask = df_all["game index"].isin(user_info["Rated games index"].item())
    df_all.loc[mask, "color"] = df_all.loc[mask, "color"].apply(lambda _: [0, 0, 255])

    # Top 5 games are rose (fuxia)
    mask = df_all["game index"].isin(user_info["Top games"].item())
    df_all.loc[mask, "color"] = df_all.loc[mask, "color"].apply(lambda _: [255, 0, 161])

    points = df_all[df_all["game index"].isin(plotted_games_index)]
    points["name"] = np.round(np.clip(nmf_pred[user_index, plotted_games_index]
                              * 8 + 2, 0, 10), 1)  # Predicted rating on hover

    view = recalc_view(points, games_info, initial_view_state)
    new_deck = get_deck(points, view, compute_point_size(points.shape[0]))

    return new_deck.to_json(), {"html": "<b>Note du jeu:</b> {name}"}, user_index, False, 'dropdown-reco-games', 'button-reco-games'


@app.callback(
    Output('dropdown-reco-games', 'className', allow_duplicate=True),
    Output('dropdown-reco-games', 'children'),
    Output('button-reco-games', 'className', allow_duplicate=True),
    Output('clear-dropdown-interval', 'disabled', allow_duplicate=True),

    Input('button-reco-games', 'n_clicks'),
    State('dropdown-reco-games', "className"),
    State('current-user', 'data'),
    prevent_initial_call=True
)
def show_reco_games(n_clicks, current_classname, current_user_index):
    if "open" in current_classname:  # dropdown is open -> close it
        # no_update for children allow to postpone the removal of children and hence allowing the animation
        return 'dropdown-reco-games', no_update, 'button-reco-games', False

    # TO DO : optimize for 1 search only

    # Top 5 games indices (recommended)
    user_info = users_info[users_info["User index"] == current_user_index]
    top_games = user_info["Top games"].values[0]
    print(nmf_pred[current_user_index, top_games])
    children = [_get_reco_game_div(game, rating)
                for game, rating in zip(top_games, np.clip(nmf_pred[current_user_index, top_games] * 8 + 2, 0, 10))]

    return 'dropdown-reco-games open', children, 'button-reco-games open', True


@app.callback(
    Output('dropdown-reco-games', 'children', allow_duplicate=True),
    Output('clear-dropdown-interval', 'disabled', allow_duplicate=True),
    Input('clear-dropdown-interval', 'n_intervals'),
    prevent_initial_call=True
)
def clear_children(n_intervals):
    return [], True


def _get_reco_game_div(game, rating):
    game_info = games_info[games_info["Game index"] == game]

    return html.Div(html.Button([html.Div(html.B(game_info["Game name year"].item()), style={"textAlign": "left", "width": "80%", "overflowX": "hidden"}),
                                 html.Div([html.B(f"{rating:.1f}"), " ⭐"], style={"textAlign": "right", "width": "20%"})],
                                style={"border": "none", "backgroundColor": "#ffffff",
                                       "display": "flex", "width": "100%"},
                                id={"type": "reco-game", "index": game.item()}, n_clicks=0),
                    className="item-reco-games", style={"backgroundColor": "#ffffff",
                                                        "padding": "10px",
                                                        "display": "flex",
                                                        "margin": "0px 20px 10px 20px",
                                                        "justifyContent": "spaceBetween",
                                                        "alignItems": "center",
                                                        "borderRadius": "10px",
                                                        "width": "auto"})


@app.callback(
    Output('current-cluster', 'data', allow_duplicate=True),
    Output('tsne', 'data', allow_duplicate=True),
    Output('plotted-data', 'data'),
    Input({'type': 'reco-game', 'index': ALL}, 'n_clicks'),
    State('plotted-data', 'data'),
    State('current-cluster', 'data'),
    prevent_initial_call=True
)
def go_to_point(n_clicks, plotted_games_index, current_cluster):

    # Prevent initial update
    if not any(n > 0 for n in n_clicks):
        return no_update, no_update, no_update

    if ctx.triggered_id is None:
        return no_update, no_update, no_update

    game_info = df_all[df_all["game index"] == ctx.triggered_id["index"]].iloc[0]
    selected_cluster = game_info["cluster"].item()

    if selected_cluster == current_cluster:
        return no_update, no_update, no_update

    points = df_all[df_all["cluster"] == selected_cluster]

    lat, lon, target = get_view_params(points)
    view_state = pdk.ViewState(latitude=lat, longitude=lon, target=target,
                               controller=True, rotation_x=0, rotation_orbit=0, zoom=4)

    new_deck = get_deck(points, view_state, compute_point_size(points.shape[0]))
    return selected_cluster, new_deck.to_json(), points["game index"].to_list()


if __name__ == '__main__':
    app.run(debug=True)
