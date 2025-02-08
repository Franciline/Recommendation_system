import pandas as pd
import numpy as np
from scipy.sparse import dok_array, csr_array
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import mean_squared_error
from .CF_knn import get_KNN, predict_ratings_baseline
# BASELINE


def hide_ratings(matrix_ratings: dok_array, mask_ratings: dok_array, user_ind) -> np.array:
    # Separate games into train, test dataset

    # indices of games that user has rated
    games_ind = mask_ratings[user_ind].nonzero()[0]
    # hide 30% of his games
    games_to_hide = np.random.choice(games_ind, int(0.3 * games_ind.size), replace=False)
    # games_preserved = np.setdiff1d(games_ind, games_to_hide, assume_unique=True)

    # Hide ratings
    matrix_ratings[user_ind, games_to_hide] = 0
    mask_ratings[user_ind, games_to_hide] = 0

    # np.nanmean is not used to ignore zeroes (not rated games)
    # matrix_ratings_train.eliminate_zeros()
    # mask_ratings_train.eliminate_zeros()
    return games_to_hide


# COSINE ONLY
def recalc_cos_similarity(matrix_ratings: dok_array, similarity_matrix: dok_array, user_ind) -> None:
    new_distance = cosine_distances(matrix_ratings[user_ind].reshape(1, -1), matrix_ratings)[0]
    # for user in range(matrix_ratings.shape[0]):
    similarity_matrix[:, user_ind] = new_distance
    similarity_matrix[user_ind, :] = new_distance


def calc_RMSE_cos(user: int, matrix_ratings: csr_array, mask_ratings: csr_array, similarity_matrix: np.ndarray) -> float:
    # Conversion for efficient modification
    dok_ratings = matrix_ratings.tolil()
    dok_mask_ratings = mask_ratings.tolil()

    if not isinstance(similarity_matrix, np.ndarray):
        dok_similarity = similarity_matrix.todok()
    else:
        dok_similarity = similarity_matrix

    rmse = np.nan  # = nan if cannot be predicted
    # hide games, create new user_game_matrix
    # & memorise to restore later
    old_user_ratings = matrix_ratings[user].toarray()
    old_user_mask_ratings = mask_ratings[user].toarray()
    # modify dok ratings, dok_mask_ratings
    hidden_games = hide_ratings(dok_ratings, dok_mask_ratings, user)
    # modify dok_similarity for this row
    old_similarity_row = similarity_matrix[user]
    recalc_cos_similarity(dok_ratings, dok_similarity, user)
    # find similar users
    similar_users = get_KNN(dok_similarity, k=35, user_ind=user, dtype="cos")
    # predict ratings (if possible)
    all_ratings, predicted_ratings = predict_ratings_baseline(dok_ratings, dok_mask_ratings, similar_users)

    # calc rmse
    to_eval = np.intersect1d(predicted_ratings, hidden_games)
    if to_eval.size != 0:
        rmse = mean_squared_error(matrix_ratings[user, to_eval].toarray(), all_ratings[to_eval])

    # Update dok_ratings and dok_mask_ratings in one operation
    dok_ratings[user] = old_user_ratings
    dok_mask_ratings[user] = old_user_mask_ratings

    # Update dok_similarity for the whole row and column in one operation
    dok_similarity[user, :] = old_similarity_row
    dok_similarity[:, user] = old_similarity_row

    # check if everything was restored
    assert ((similarity_matrix == dok_similarity).all())
    assert ((matrix_ratings.toarray() == dok_ratings.toarray()).all())
    assert ((mask_ratings.toarray() == dok_mask_ratings.toarray()).all())

    return rmse
