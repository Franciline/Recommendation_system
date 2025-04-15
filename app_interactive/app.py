# Requires Dash 2.17.0 or later

from plotly.colors import qualitative
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px  # interactivity
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# import pydeck as pdk
# import dash_deck

n_clusters = 30  # -> 30 colors to generate
offset = 200
stretch = [1., 1., 1.]
games_tsne = np.load("tsne_pushed.npy", mmap_mode="r") * stretch

clusters = np.load("clusters.npy", mmap_mode="r")

colors = np.array((qualitative.Plotly + qualitative.Set3 + qualitative.Alphabet)[:n_clusters])

# for PyDeck
# colors = [[159, 72, 203] for _ in range(games_tsne.shape[0])]
# games_tsne = pd.DataFrame(data={
#     "coordinates": games_tsne.tolist(),
#     "color": colors
# })

# view_state = pdk.ViewState(
#     latitude=-25, longitude=-17, zoom=2, pitch=45, bearing=0
# )

# arc_layer = pdk.Layer(
#     "PointCloudLayer",
#     data=games_tsne,
#     get_position="coordinates",
#     get_color="color",
#     pickable=True,
# )

# deck = pdk.Deck(arc_layer, initial_view_state=view_state)

# app.layout = html.Div(
#     dash_deck.DeckGL(
#         deck.to_json(), id="deck-gl"
#     )
# )


app = Dash()


app.layout = html.Div([
    dcc.Graph(
        id='3d-scatter',
        figure=go.Figure(
            data=[go.Scatter3d(
                x=games_tsne[:, 2],
                y=games_tsne[:, 1],
                z=games_tsne[:, 0],
                mode='markers',
                marker=dict(size=4, color=colors[clusters], opacity=0.7),
            )],
            layout=go.Layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    bgcolor='white',
                ),
            )
        ),
        style={'height': '100vh', 'width': '100vw'}
    )
], style={'margin': '0', 'padding': '0', 'height': '100vh', 'width': '100vw', 'overflow': 'hidden'})

if __name__ == '__main__':
    app.run(debug=True)
