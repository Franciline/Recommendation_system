import pandas as pd
import numpy as np
from scipy.sparse import dok_array, csr_array
from sklearn.metrics.pairwise import cosine_distances, nan_euclidean_distances
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .CF_knn import get_KNN, predict_ratings_baseline
# BASELINE


def hide_ratings(matrix_ratings, mask_ratings, user_ind, replace_with=0) -> np.array:
    """
    Modify INPLACE matrix_ratings, mask_ratings.
    Hide 30% of rated games by 'user_ind', 70% percent are not hidden.
    Values in matrix_ratings are replaced with 'replace_with' and with 0 in 'mask_ratings'.

    Parameters
    ----------
        matrix_ratings : sparse matrix
            User-game matrix
        mask_ratings : sparse matrix
            Mask matrix for matrix_ratings to indicate 1 for true rating, 0 for non existent.
        user_ind : int
            User index for who we predict ratings.
        replace_with
            Value for replacing in 'matrix_ratings'.

    Returns
    -------
        np.array : indices of hidden games
    """

    # Separate games into train, test dataset

    # indices of games that user has rated
    games_ind = mask_ratings[user_ind].nonzero()[0]
    # hide 30% of his games
    games_to_hide = np.random.choice(games_ind, int(0.3 * games_ind.size), replace=False)
    # games_preserved = np.setdiff1d(games_ind, games_to_hide, assume_unique=True)

    # Hide ratings
    matrix_ratings[user_ind, games_to_hide] = replace_with
    mask_ratings[user_ind, games_to_hide] = 0

    # np.nanmean is not used to ignore zeroes (not rated games)
    # matrix_ratings_train.eliminate_zeros()
    # mask_ratings_train.eliminate_zeros()
    return games_to_hide


# COSINE ONLY
def recalc_cos_similarity(matrix_ratings, similarity_matrix: np.ndarray, user_ind: int) -> None:
    """
    Recalculate cosine similarity matrix only for user_ind row, column.
    Modify INPLACE similarity_matrix (only column and row indicated by 'user ind')

    Parameters
    ----------
        matrix_ratings : sparse matrix
            User-game matrix
        similarity_matrix : np.ndarray
            Similarity matrix between users based on cosine distance.
        User_ind : int
            User index matrices.
    """

    new_distance = cosine_distances(matrix_ratings[user_ind].reshape(1, -1), matrix_ratings)[0]
    # for user in range(matrix_ratings.shape[0]):
    similarity_matrix[:, user_ind] = new_distance
    similarity_matrix[user_ind, :] = new_distance


def calc_RMSE_cos(user: int, matrix_ratings: csr_array, mask_ratings: csr_array, similarity_matrix: np.ndarray) -> float:
    """Calculate RMSE for 'user'. 
    To calculate RMSE:
        - hide 30% of games that 'user' rated
        - recalc similarity matrix (only rows, cols where user (=vector) participates)
        - find similar users for 'user'
        - predict ratings for games (try to predict for all, but if not possible, i.e. no similar user has rated this game, 
            then 0 is given as a rating)
        - rmse is calculated only on these hidden 30% of games & ratings also could be predicted.

    Parameters
    ----------
        user : int
            User index in matrices
        matrix_ratings: sparse matrix
            User-game matrix.
        mask_ratings : sparse matrix
            Mask matrix for matrix_ratings to indicate 1 for true rating, 0 for non existent.
        similarity_matrix : np.ndarray
            Similarity matrix between users based on cosine distance.
    Returns
    -------
        float : RMSE calculated for 'user'.
    """

    # Conversion for efficient modification
    dok_ratings = matrix_ratings.tolil()
    dok_mask_ratings = mask_ratings.tolil()

    dok_similarity = similarity_matrix

    rmse = np.nan  # = nan if cannot be predicted
    # hide games, create new user_game_matrix
    # & memorise to restore later
    old_user_ratings = matrix_ratings[user].toarray()
    old_user_mask_ratings = mask_ratings[user].toarray()
    # modify dok ratings, dok_mask_ratings
    hidden_games = hide_ratings(dok_ratings, dok_mask_ratings, user)

    # modify dok_similarity for this row
    old_similarity_row = similarity_matrix[user].copy()
    recalc_cos_similarity(dok_ratings, dok_similarity, user)
    # find similar users
    similar_users = get_KNN(dok_similarity, k=int(np.sqrt(matrix_ratings.shape[0])), user_ind=user, dtype="cos")
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


def recalc_eucl_similarity(nan_ratings, matrix_ratings, mask_ratings, similarity_matrix, mask_sim_matrix, user_ind: int) -> None:
    matrix_ratings[user_ind, ~(mask_ratings[user_ind] == 1).toarray()] = np.nan

    new_distance = nan_euclidean_distances(
        matrix_ratings[user_ind].toarray().reshape(1, -1), nan_ratings)[0]
    nans = np.isnan(new_distance)
    new_distance[nans] = 0
    new_mask = np.where(nans, 0, 1)

    similarity_matrix[user_ind, :] = new_distance
    similarity_matrix[:, user_ind] = new_distance

    mask_sim_matrix[user_ind, :] = new_mask
    mask_sim_matrix[:, user_ind] = new_mask


def calc_RMSE_eucl(user: int, matrix_ratings: csr_array, mask_ratings: csr_array, similarity_matrix: csr_array, mask_sim_matrix: csr_array) -> float:
    # Conversion for efficient modification
    inf_ratings = matrix_ratings.tolil()
    nan_ratings = matrix_ratings.toarray()
    nan_ratings[~(mask_ratings == 1).toarray()] = np.nan

    lil_mask_ratings = mask_ratings.tolil()

    dok_similarity = similarity_matrix.todok()

    dok_mask_sim = mask_sim_matrix.todok()

    rmse = np.nan  # = nan if cannot be predicted
    # hide games, create new user_game_matrix
    # & memorise to restore later
    old_user_ratings = matrix_ratings[user].copy()
    old_user_mask_ratings = mask_ratings[user].toarray()

    # modify dok ratings, dok_mask_ratings

    hidden_games = hide_ratings(inf_ratings, lil_mask_ratings, user, replace_with=np.nan)
    # modify dok_similarity for this row
    old_similarity_row = similarity_matrix[user].toarray()
    old_similarity_col = similarity_matrix[:, user].toarray()
    old_mask_sim_row = mask_sim_matrix[user].toarray()

    recalc_eucl_similarity(nan_ratings, inf_ratings, lil_mask_ratings, dok_similarity, dok_mask_sim, user)
    # find similar users
    similar_users = get_KNN(dok_similarity, k=int(
        np.sqrt(matrix_ratings.shape[0])), user_ind=user, dtype="euclidean", mask_sim_matrix=dok_mask_sim)
    # predict ratings (if possible)
    all_ratings, predicted_ratings = predict_ratings_baseline(inf_ratings, lil_mask_ratings, similar_users)

    # calc rmse
    to_eval = np.intersect1d(predicted_ratings, hidden_games)

    if to_eval.size != 0:
        rmse = mean_squared_error(matrix_ratings[user, to_eval].toarray(), all_ratings[to_eval])
        # print(matrix_ratings[user, to_eval].toarray(), inf_ratings[user, to_eval].toarray(), all_ratings[to_eval])
        # Update dok_ratings and dok_mask_ratings in one operation
    inf_ratings[user] = old_user_ratings
    lil_mask_ratings[user] = old_user_mask_ratings

    # Update dok_similarity for the whole row and column in one operation

    dok_similarity[user, :] = old_similarity_row
    dok_similarity[:, user] = old_similarity_col
    # print(dok_similarity[:, user].shape, old_similarity_row.T.shape)
    dok_mask_sim[user, :] = old_mask_sim_row
    dok_mask_sim[:, user] = old_mask_sim_row.T

    # check if everything was restored

    # assert((similarity_matrix.toarray() == dok_similarity.toarray()).all())

    # assert ((matrix_ratings.toarray() == inf_ratings.toarray()).all())
    # assert ((mask_ratings.toarray() == lil_mask_ratings.toarray()).all())
    # assert ((mask_sim_matrix.toarray() == dok_mask_sim.toarray()).all())

    return rmse

# calculate MAE


def calc_MAE_cos(user: int, matrix_ratings: csr_array, mask_ratings: csr_array, similarity_matrix: np.ndarray) -> float:
    """Calculate MAE for 'user'. 
    To calculate MAE:
        - hide 30% of games that 'user' rated
        - recalc similarity matrix (only rows, cols where user (=vector) participates)
        - find similar users for 'user'
        - predict ratings for games (try to predict for all, but if not possible, i.e. no similar user has rated this game, 
            then 0 is given as a rating)
        - mae is calculated only on these hidden 30% of games & ratings also could be predicted.

    Parameters
    ----------
        user : int
            User index in matrices
        matrix_ratings: sparse matrix
            User-game matrix.
        mask_ratings : sparse matrix
            Mask matrix for matrix_ratings to indicate 1 for true rating, 0 for non existent.
        similarity_matrix : np.ndarray
            Similarity matrix between users based on cosine distance.
    Returns
    -------
        float : MAE calculated for 'user'.
    """

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
    old_similarity_row = similarity_matrix[user].copy()
    recalc_cos_similarity(dok_ratings, dok_similarity, user)
    # find similar users
    similar_users = get_KNN(dok_similarity, k=int(np.sqrt(matrix_ratings.shape[0])), user_ind=user, dtype="cos")
    # predict ratings (if possible)
    all_ratings, predicted_ratings = predict_ratings_baseline(dok_ratings, dok_mask_ratings, similar_users)

    # calc rmse
    to_eval = np.intersect1d(predicted_ratings, hidden_games)

    if to_eval.size != 0:
        mae = mean_absolute_error(matrix_ratings[user, to_eval].toarray(), all_ratings[to_eval])

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

    return mae
