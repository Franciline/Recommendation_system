"""Neighbor-selection helpers for bigram evaluation."""

import numpy as np

from boardgames_recsys.evaluation.ratings import hide_ratings, recalc_cos_similarity
from boardgames_recsys.models.collaborative_filtering import get_KNN, predict_ratings_baseline


def _knn_sim_neg_pos(
    user_id,
    games_to_consider,
    matrix_ratings,
    mask_ratings,
    cos_sim_matrix,
    users_table,
    games_table,
    comments_all,
    users_mean,
    type="simi",
    k=40,
):
    """Return comments of a user and neighbors for predicted positive/negative games."""

    user_ind = users_table[users_table == user_id].index[0]
    games_to_hide = np.random.choice(games_to_consider, size=200, replace=False)

    hidden_games = np.intersect1d(
        games_table[games_table.isin(games_to_hide)].index, mask_ratings[user_ind, :].nonzero()[0]
    )

    prev_ratings, prev_mask_ratings = matrix_ratings[user_ind, :], mask_ratings[user_ind, :]
    prev_sim = cos_sim_matrix[user_ind, :]

    matrix_ratings[user_ind, hidden_games] = 0
    mask_ratings[user_ind, hidden_games] = 0

    recalc_cos_similarity(user_ind, matrix_ratings, cos_sim_matrix)

    knn_all_user = get_KNN(cos_sim_matrix, users_table.shape[0], user_ind)

    match type:
        case "simi":
            sim_users = knn_all_user[:k]
        case "less_simi":
            sim_users = knn_all_user[-k:]
        case "random":
            sim_users = np.random.choice(knn_all_user, size=k, replace=False)

    pred_ratings, mask_pred_ratings = predict_ratings_baseline(
        matrix_ratings, mask_ratings, sim_users, cos_sim_matrix, user_ind
    )

    matrix_ratings[user_ind, :], mask_ratings[user_ind, :] = prev_ratings, prev_mask_ratings
    cos_sim_matrix[user_ind, :], cos_sim_matrix[:, user_ind] = prev_sim, prev_sim

    diff = np.abs(matrix_ratings[user_ind, hidden_games] - pred_ratings[hidden_games])

    allow_err = 2
    user_mean = users_mean.loc[users_mean["User id"] == user_id, "Rating"].item()
    pos, neg = pred_ratings[hidden_games] < user_mean, pred_ratings[hidden_games] > user_mean

    neg_pred_games = hidden_games[np.argwhere(neg & (diff < allow_err)).flatten()]
    pos_pred_games = hidden_games[np.argwhere(pos & (diff < allow_err)).flatten()]

    neg_pred_games = games_table[games_table.index.isin(neg_pred_games)].values
    pos_pred_games = games_table[games_table.index.isin(pos_pred_games)].values

    sim_users = users_table[users_table.index.isin(sim_users)].values
    sim_users_neg = comments_all[comments_all["Game id"].isin(neg_pred_games) & comments_all["User id"].isin(sim_users)]
    sim_users_pos = comments_all[comments_all["Game id"].isin(pos_pred_games) & comments_all["User id"].isin(sim_users)]

    user_pos = comments_all[(comments_all["Game id"].isin(neg_pred_games)) & (comments_all["User id"] == user_id)]
    user_neg = comments_all[(comments_all["Game id"].isin(pos_pred_games)) & (comments_all["User id"] == user_id)]

    return sim_users_neg, sim_users_pos, user_neg, user_pos


def _knn_sim(user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, k=40):
    """Return well-predicted game IDs and sorted nearest-neighbor user IDs."""

    user_ind = users_table[users_table == user_id].index[0]

    prev_ratings, prev_mask_ratings = matrix_ratings[user_ind, :], mask_ratings[user_ind, :]
    prev_sim = cos_sim_matrix[user_ind, :]

    hidden_games = hide_ratings(matrix_ratings, mask_ratings, user_ind, 0.1)
    recalc_cos_similarity(user_ind, matrix_ratings, cos_sim_matrix)

    knn_all_user = get_KNN(cos_sim_matrix, users_table.shape[0], user_ind)

    pred_ratings, mask_pred_ratings = predict_ratings_baseline(
        matrix_ratings, mask_ratings, knn_all_user[:k], cos_sim_matrix, user_ind
    )
    hidden_games = np.intersect1d(hidden_games, mask_pred_ratings)

    matrix_ratings[user_ind, :], mask_ratings[user_ind, :] = prev_ratings, prev_mask_ratings
    cos_sim_matrix[user_ind, :], cos_sim_matrix[:, user_ind] = prev_sim, prev_sim

    allow_err = 2
    diff = np.abs(matrix_ratings[user_ind, hidden_games] - pred_ratings[hidden_games])

    well_predicted_games = hidden_games[np.argwhere(diff < allow_err).flatten()]
    well_predicted_games = games_table[games_table.index.isin(well_predicted_games)].values

    return well_predicted_games, users_table.loc[knn_all_user].values
