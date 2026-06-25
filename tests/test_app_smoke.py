from boardgames_recsys.app.app import APP_DATA_DIR, app, games_info, get_user_tsne, users_info


def test_app_imports_with_default_data():
    assert APP_DATA_DIR.exists()
    assert app.layout is not None
    assert not games_info.empty


def test_get_user_tsne_returns_one_value_per_callback_output():
    plotted_games = games_info["game index"].head(10).to_list()
    user_index = int(users_info["User index"].iloc[0])

    assert len(get_user_tsne("", plotted_games, -2)) == 7
    assert len(get_user_tsne(user_index, plotted_games, -2)) == 7
