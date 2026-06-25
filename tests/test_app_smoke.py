from boardgames_recsys.app.app import APP_DATA_DIR, app, games_info


def test_app_imports_with_default_data():
    assert APP_DATA_DIR.exists()
    assert app.layout is not None
    assert not games_info.empty
