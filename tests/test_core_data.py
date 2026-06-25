import pandas as pd

from boardgames_recsys.filter import filter_df
from boardgames_recsys.user_game_matrix import get_matrix_user_game


def test_filter_df_removes_sparse_users_and_games():
    reviews = pd.DataFrame(
        [
            {"User id": 1, "Game id": 10, "Rating": 8},
            {"User id": 1, "Game id": 20, "Rating": 7},
            {"User id": 2, "Game id": 10, "Rating": 6},
            {"User id": 2, "Game id": 20, "Rating": 9},
            {"User id": 3, "Game id": 30, "Rating": 5},
        ]
    )

    filtered = filter_df(reviews, min_reviews=2)

    assert set(filtered["User id"]) == {1, 2}
    assert set(filtered["Game id"]) == {10, 20}
    assert len(filtered) == 4


def test_get_matrix_user_game_returns_ratings_mask_and_associations():
    reviews = pd.DataFrame(
        [
            {"User id": 2, "Game id": 20, "Rating": 9},
            {"User id": 1, "Game id": 10, "Rating": 8},
            {"User id": 1, "Game id": 20, "Rating": 7},
        ]
    )

    ratings, mask, users, games = get_matrix_user_game(reviews)

    assert ratings.shape == (2, 2)
    assert mask.shape == (2, 2)
    assert users.tolist() == [1, 2]
    assert games.tolist() == [10, 20]
    assert ratings.toarray().tolist() == [[8, 7], [0, 9]]
    assert mask.toarray().tolist() == [[1, 1], [0, 1]]
