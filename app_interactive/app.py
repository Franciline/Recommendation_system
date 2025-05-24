from math import log2
import os
import dash_bootstrap_components as dbc
from plotly.colors import qualitative
from dash import Dash, html, dcc, callback, Output, Input, State, ctx, ALL, MATCH, no_update
import pandas as pd
import numpy as np
import pydeck as pdk
import dash_deck
from dash_extensions import Lottie


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


def compute_zoom(points, n_points, min_points=150, max_points=400, min_zoom_level=0, max_zoom_level=4):
    range_x = points["x"].max() - points["x"].min()
    range_y = points["y"].max() - points["y"].min()
    range_z = points["z"].max() - points["z"].min()
    max_range = max(range_x, range_y, range_z)

    if max_range == 0:
        return 16
    zoom = log2(350 / max_range)
    return max(min(zoom, 5), 0)


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
                             controller=True, rotation_x=0, rotation_orbit=0, zoom=compute_zoom(points, points.shape[0]))
    return view


n_clusters = 30  # -> 30 colors to generate
games_tsne = np.load("tsne_pushed.npy", mmap_mode="r")   # TSNE 3D
clusters = np.load("clusters.npy", mmap_mode="r")        # Clusters assignment

# NNMF prediction U @ G.T (already existing ratings are replaced by true ones)
nmf_pred = np.load("nnmf_prediction.npy", mmap_mode="r")

# users_info = pd.read_parquet("users_info.parquet")  # info on users
users_info = pd.read_json("users_info.json", orient="records")
games_info = pd.read_json("games_info.json", orient="records")

# For themes-dropdown
themes = {
    "Tous les clusters": [],
    "🧩🕵️ Logique et Déduction": [22, 23],
    "🏗️🏰 Construction & expansion": [0, 19],
    "🧠⚡ Rapide & Tactique": [14, 20],
    "📚⏳ Longs & complexes": [2, 9, 13, 25, 29],
    "💎🃏 Collecte":  [1, 17, 26],
    "🪖💣 Guerre": [5],
    "🏛️🎲 Eurogames": [15, 16],
    "🌍🏺 Civilisation": [12, 18, 21],
    "🗡️🚩 Capture territoire": [3, 11, 27],
    "🌅🖼️ Superbes visuels": [4, 8],
    "🐉📜 Culture | Fantaisie": [6, 7, 28],
    "👎 Coups de blues": [10],
    "🏆 Coups de coeurs": [24],
}
themes_inverse = {cluster: theme for theme, clusters in themes.items() for cluster in clusters}
# themes_names = games_info["name"]
colors_points = games_info["color"]

# For animation fireworks
fw_hidden = dict(loop=False, autoplay=False,
                 style={"display": "none", "position": "absolute", "left": "15%", "top": "10%"})
fw_shown = dict(loop=False, autoplay=True,
                style={"display": "block", "position": "absolute", "left": "15%", "top": "10%"})
special_index = 39  # index

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# View
lat, lon, target = get_view_params(games_info)
initial_view_state = pdk.ViewState(latitude=lat, longitude=lon, target=target,
                                   controller=True, rotation_x=15, rotation_orbit=30, zoom=-0.2)

deck = get_deck(games_info, initial_view_state)


# Main layout : TSNE (DeckGL) on the left, sidebar on the right
app.layout = html.Div([
    dcc.Store(id="explore-mode", data=True),  # True if exploration mode, False if recommendation mode
    dcc.Store(id="plotted-data", data=games_info["game index"].values),  # store plotted games indices

    # allows not to change cluster if only one cluster shown. >= 0 if one cluster. -1 if thematic cluster. -2 if all clusters
    dcc.Store(id="current-cluster", data=None),

    # makes sense in reco mode. Index of current chosen user. Necessary to show recommended games
    dcc.Store(id="current-user", data=None),

    # interval to allow animation of closing dropdown-reco-games
    dcc.Interval(id='clear-dropdown-interval', interval=400, n_intervals=0, disabled=True),
    dcc.Interval(id='delete-animation', interval=2800, n_intervals=0, disabled=True),

    html.Div([
        # Header when only one cluster is shown

        html.Div(children="Tous les clusters", id="cluster-theme-header", className='cluster-theme-header'),
        Lottie(id="fireworks", options=fw_hidden,
               url="https://lottie.host/7be84abc-372f-4d26-9794-96ba22ca6f6d/Cx1QOvmfIU.json"),
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
                             value='Tous les clusters', id='themes-dropdown', searchable=False, clearable=False, className="rounded-dropdown",
                             style={"width": "100%", "flex": "1"})
            ], style={"padding": "20px", "display": "flex", "width": "100%", 'alignItems': 'center', "height": "fitContent"}),

            html.Div(children=[dcc.Dropdown(options=[{"label": "Choisir un profil", "value": "..."}] +
                                            [{"label": f"User {username}", "value": index}
                                             if index != special_index else {"label": f"🎉User {username}🎉", "value": index}
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
            "overflowY": "scroll",
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
    State('explore-mode', 'data'),
    prevent_initial_call=True,
)
def thematic_clusters(value, explore_mode):
    """Plot only selected thematic clusters."""
    global games_info

    if value is None:
        return (no_update,) * 4

    current_cluster = -1
    if value == "Tous les clusters":
        points = games_info
        view_state = initial_view_state
        current_cluster = -2

    else:
        points = games_info[games_info["cluster"].isin(themes[value])]

        if explore_mode:
            points["name"] = points["game name year"]

        view_state = recalc_view(points, games_info, initial_view_state)

        # target = [points.x.mean(), points.y.mean(), points.z.mean()]
        # lat, lon, target = get_view_params(points)

        # view_state = pdk.ViewState(latitude=lat, longitude=lon, target=target,
        #                            controller=True, rotation_x=0, rotation_orbit=0, zoom=1.5)

    new_deck = get_deck(points, view_state, compute_point_size(points.shape[0]))
    return new_deck.to_json(), points["game index"].values, current_cluster, value


# Click on a cluster -> zoom on this cluster, hide clusters points


@app.callback(
    Output('themes-dropdown', 'value'),
    Output('tsne', 'data', allow_duplicate=True),
    Output("plotted-data", "data", allow_duplicate=True),
    Output('current-cluster', 'data', allow_duplicate=True),
    Output('cluster-theme-header', 'children', allow_duplicate=True),

    Input('tsne', 'clickInfo'),
    State('current-cluster', 'data'),
    State('explore-mode', 'data'),
    prevent_initial_call=True
)
def zoom_cluster(clickInfo, current_cluster, explore_mode):
    """Plot only one cluster on a click on one of its points"""
    if (clickInfo is None) or ("object" not in clickInfo) or clickInfo["object"] is None:
        return (no_update,) * 5

    clicked_cluster = clickInfo["object"]["cluster"]
    if current_cluster == clicked_cluster:
        return (no_update,) * 5

    points = games_info[games_info["cluster"] == clicked_cluster]

    if explore_mode:
        points["name"] = points["game name year"]

    view_state = recalc_view(points, games_info, initial_view_state)
    # lat, lon, target = get_view_params(points)
    # view_state = pdk.ViewState(latitude=lat, longitude=lon, target=target,
    #                            controller=True, rotation_x=0, rotation_orbit=0, zoom=4)

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
    if click_info is None:
        return [], {"display": "none"},

    if click_info["object"] is None:
        return [], {"display": "none"}

    game_info = click_info["object"]

    return [html.Div([html.H4(html.B(game_info["game name year"]), style={"margin-bottom": "20px", "width": "80%"}),
                      html.H3([f"{game_info['rating']:.1f}", " ⭐"], style={"width": "20%", "textAlign": "right"})],
                     style={"display": "flex", "justifyContent": "spaceBetween"}),
            html.P(html.P([html.B("🎲 Type: "), game_info['type']])),
            html.Div([html.P([html.B("👨‍👩‍👧‍👦 Joueurs: "), game_info['players']]),
                      html.P([html.B("🎂 Age: "), game_info['age']])],


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
    Output('fireworks', 'options', allow_duplicate=True),

    Input('mode-button', 'n_clicks'),  # n_clicks is not used, but Dash demands non empty property
    State('explore-mode', 'data'),
    State('plotted-data', 'data'),
    State('current-user', 'data'),
    prevent_initial_call=True
)
def change_mode(click_info, explore_mode, plotted_games_index, current_user):
    """Change mode :
        1. recommendation -> exploration or
        2. exploration    -> recommendation

    Change of colors on df is applied only if 'recommendation -> exploration'
    """

    global themes, games_info, colors_points, initial_view_state
    if click_info is None:
        return (no_update,) * 8

    # points = games_info[games_info["game index"].isin(plotted_games_index)]
    # view = recalc_view(points, games_info, initial_view_state)
    # new_deck = get_deck(points, view, compute_point_size(points.shape[0]))

    # Exploration mode -> go to Reco mode
    if explore_mode == True:
        common = False, "...", "🔁 Exploration", False,
        if current_user == special_index:
            return *common, *(no_update,) * 3, fw_hidden
        return *common, *(no_update,) * 4

    # Reco mode -> go to Exploration mode. Dropdown : cluster themes.
    # Points will be replotted by 'get_user_tsne'. Same for changed in 'games_info'

    return True, "...", "🔁 Recommandation", True, no_update, 'dropdown-reco-games', 'button-reco-games', no_update


@app.callback(
    Output('tsne', 'data', allow_duplicate=True),
    Output('tsne', 'tooltip'),
    Output('current-user', 'data'),
    Output('button-reco-games', 'disabled'),
    Output('dropdown-reco-games', 'className'),
    Output('button-reco-games', 'className'),
    Output('fireworks', 'options', allow_duplicate=True),
    Output('delete-animation', 'disabled', allow_duplicate=True),

    Input("users-dropdown", "value"),
    State("plotted-data", "data"),
    State('explore-mode', 'data'),
    State('delete-animation', 'n_intervals'),
    State('current-cluster', 'data'),
    prevent_initial_call=True
)
def get_user_tsne(user_index, plotted_games_index, explore_mode, n_intervals, current_cluster):
    """Replot plotted games to change their color based on predicted ratings with NNMF

    Note :
        !All! colors in 'games_info' are changed since it is more logical for a user to go explore other points
    """

    global initial_view_state, games_info

    if user_index is None or plotted_games_index is None:
        return (no_update,) * 8

    points = games_info[games_info["game index"].isin(plotted_games_index)]
    view = recalc_view(points, games_info, initial_view_state)

    # print("User dropdown value changed to ", user_index)
    if user_index == "...":  # No user selected
        if explore_mode:
            games_info["name"] = games_info["theme"] if current_cluster == -2 else games_info["game name year"]

        games_info["color"] = colors_points

        points = games_info[games_info["game index"].isin(plotted_games_index)]
        view = recalc_view(points, games_info, initial_view_state)

        new_deck = get_deck(points, view, compute_point_size(points.shape[0]))
        return new_deck.to_json(), {"text": "{name}"}, no_update, True, 'dropdown-reco-games', 'button-reco-games', fw_hidden, True

    # No user change -> the event is not fired

    points = games_info[games_info["game index"].isin(plotted_games_index)]
    view = recalc_view(points, games_info, initial_view_state)

    ratings = nmf_pred[user_index, :]

    # RGB. Rating 0 = red, Rating 10 = green
    red = (255 * (1 - ratings)).astype(int)
    green = (255 * ratings).astype(int)
    blue = np.zeros_like(red)

    # Reassign colors
    colors = np.stack([red, green, blue], axis=1).tolist()
    games_info["color"] = colors

    # Color rated games to blue
    user_info = users_info[users_info["User index"] == user_index]

    mask = games_info["game index"].isin(user_info["Rated games index"].item())
    games_info.loc[mask, "color"] = games_info.loc[mask, "color"].apply(lambda _: [0, 0, 255])

    # Top 5 games are rose (fuxia)
    mask = games_info["game index"].isin(user_info["Top games"].item())
    games_info.loc[mask, "color"] = games_info.loc[mask, "color"].apply(lambda _: [255, 0, 161])

    games_info["name"] = np.round(np.clip(nmf_pred[user_index, :] * 8 + 2, 0, 10), 1)  # Predicted rating on hover

    points = games_info[games_info["game index"].isin(plotted_games_index)]

    view = recalc_view(points, games_info, initial_view_state)
    new_deck = get_deck(points, view, compute_point_size(points.shape[0]))

    common = new_deck.to_json(), {
        "html": "<b>Note du jeu:</b> {name}"}, user_index, False, 'dropdown-reco-games', 'button-reco-games'

    if user_index == special_index and n_intervals == 0:  # fireworks only on 1st clickon special user
        return *common,  fw_shown, False
    return *common, fw_hidden, True


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
    game_info = games_info[games_info["game index"] == game]

    return html.Div(html.Button([html.Div(html.B(game_info["game name year"].item()), style={"textAlign": "left", "width": "80%", "overflowX": "hidden"}),
                                 html.Div([html.B(f"{rating:.1f}"), " ⭐"], style={"textAlign": "right", "width": "20%"})],
                                style={"border": "none", "backgroundColor": "#ffffff",
                                       "display": "flex", "width": "100%"},
                                id={"type": "reco-game", "index": game_info["game index"].item()}, n_clicks=0),
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
    if not any(n > 0 for n in n_clicks) or ctx.triggered_id is None:
        return (no_update,) * 3

    game_info = games_info[games_info["game index"] == ctx.triggered_id["index"]].iloc[0]
    selected_cluster = game_info["cluster"].item()

    if selected_cluster == current_cluster:
        return (no_update,) * 3

    points = games_info[games_info["cluster"] == selected_cluster]
    view_state = recalc_view(points, games_info, initial_view_state)
    # lat, lon, target = get_view_params(points)
    # view_state = pdk.ViewState(latitude=lat, longitude=lon, target=target,
    #                            controller=True, rotation_x=0, rotation_orbit=0, zoom=4)

    new_deck = get_deck(points, view_state, compute_point_size(points.shape[0]))
    return selected_cluster, new_deck.to_json(), points["game index"].to_list()


@app.callback(
    Output('delete-animation', 'disabled', allow_duplicate=True),
    Output('fireworks', 'options', allow_duplicate=True),
    Input('delete-animation', 'n_intervals'),
    prevent_initial_call=True
)
def delete_animation(n_intervals):
    print("delete", n_intervals)
    return True, fw_hidden


if __name__ == '__main__':
    app.run(debug=True)
