from scipy.sparse import csr_array
import pandas as pd
import numpy as np
from scipy.sparse import csr_array, lil_array
from sklearn.metrics.pairwise import cosine_distances, nan_euclidean_distances
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from .CF_knn import get_KNN, predict_ratings_baseline


def hide_ratings(matrix_ratings, mask_ratings, user_ind, percentage: float = 0.3) -> np.array:
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
        percentage: float
            Percentage of games to hide

    Returns
    -------
        np.array : indices of hidden games
    """

    # Separate games into train, test dataset

    # indices of games that user has rated
    games_ind = mask_ratings[user_ind].nonzero()[0]
    # hide 30% of his games
    games_to_hide = np.random.choice(games_ind, int(percentage * games_ind.size), replace=False)
    # games_preserved = np.setdiff1d(games_ind, games_to_hide, assume_unique=True)

    # Hide ratings
    matrix_ratings[user_ind, games_to_hide] = 0
    mask_ratings[user_ind, games_to_hide] = 0

    # np.nanmean is not used to ignore zeroes (not rated games)
    # matrix_ratings_train.eliminate_zeros() -> cannot cause ratings with 0 values exists
    # mask_ratings_train.eliminate_zeros()
    return games_to_hide


def recalc_cos_similarity(user_ind: int, matrix_ratings, similarity_matrix: np.ndarray) -> None:
    """
    Recalculate cosine similarity matrix only for user_ind row, column.
    Modify INPLACE similarity_matrix (only column and row indicated by 'user ind')

    Parameters
    ----------
        user_ind : int
            User index matrices.
        matrix_ratings : sparse matrix
            User-game ratings matrix
        similarity_matrix : np.ndarray
            Similarity matrix between users based on cosine distance.
    """

    new_distance = cosine_distances(matrix_ratings[user_ind].reshape(1, -1), matrix_ratings)[0]
    # for user in range(matrix_ratings.shape[0]):
    similarity_matrix[:, user_ind] = new_distance
    similarity_matrix[user_ind, :] = new_distance


# def calc_RMSE_cos(user: int, matrix_ratings: csr_array, mask_ratings: csr_array, similarity_matrix: np.ndarray) -> float:
#     """Calculate RMSE for 'user'.
#     To calculate RMSE:
#         - hide 30% of games that 'user' rated
#         - recalc similarity matrix (only rows, cols where user (=vector) participates)
#         - find similar users for 'user'
#         - predict ratings for games (try to predict for all, but if not possible, i.e. no similar user has rated this game,
#             then 0 is given as a rating)
#         - rmse is calculated only on these hidden 30% of games & ratings also could be predicted.

#     Parameters
#     ----------
#         user : int
#             User index in matrices
#         matrix_ratings: sparse matrix
#             User-game matrix.
#         mask_ratings : sparse matrix
#             Mask matrix for matrix_ratings to indicate 1 for true rating, 0 for non existent.
#         similarity_matrix : np.ndarray
#             Similarity matrix between users based on cosine distance.
#     Returns
#     -------
#         float : RMSE calculated for 'user'.
#     """

#     # Conversion for efficient modification
#     dok_ratings = matrix_ratings.tolil()
#     dok_mask_ratings = mask_ratings.tolil()

#     dok_similarity = similarity_matrix

#     rmse = np.nan  # = nan if cannot be predicted
#     # hide games, create new user_game_matrix
#     # & memorise to restore later
#     old_user_ratings = matrix_ratings[user].toarray()
#     old_user_mask_ratings = mask_ratings[user].toarray()
#     # modify dok ratings, dok_mask_ratings
#     hidden_games = hide_ratings(dok_ratings, dok_mask_ratings, user)

#     # modify dok_similarity for this row
#     old_similarity_row = similarity_matrix[user].copy()
#     recalc_cos_similarity(dok_ratings, dok_similarity, user)
#     # find similar users
#     similar_users = get_KNN(dok_similarity, k=int(np.sqrt(matrix_ratings.shape[0])), user_ind=user)
#     # predict ratings (if possible)
#     all_ratings, predicted_ratings = predict_ratings_baseline(dok_ratings, dok_mask_ratings, similar_users)

#     # calc rmse
#     to_eval = np.intersect1d(predicted_ratings, hidden_games)

#     if to_eval.size != 0:
#         rmse = root_mean_squared_error(matrix_ratings[user, to_eval].toarray(), all_ratings[to_eval])

#     # Update dok_ratings and dok_mask_ratings in one operation
#     dok_ratings[user] = old_user_ratings
#     dok_mask_ratings[user] = old_user_mask_ratings

#     # Update dok_similarity for the whole row and column in one operation
#     dok_similarity[user, :] = old_similarity_row
#     dok_similarity[:, user] = old_similarity_row

#     # check if everything was restored
#     assert ((similarity_matrix == dok_similarity).all())
#     assert ((matrix_ratings.toarray() == dok_ratings.toarray()).all())
#     assert ((mask_ratings.toarray() == dok_mask_ratings.toarray()).all())

#     return rmse


def recalc_eucl_similarity(user_ind: int, matrix_ratings, mask_ratings, similarity_matrix: lil_array) -> None:
    R, M = matrix_ratings, mask_ratings
    UR, UM = R[user_ind], M[user_ind]

    R_mask = R.multiply(UM)
    eucl_squared = M.dot(UR.multiply(UR)) + R_mask.dot(R_mask.T).diagonal() - 2 * R_mask.dot(UR)

    # Ponderation (division) by a number of shared games
    weights = M.dot(UM.transpose()).tocsr()
    inverse_weights = csr_array((1/weights.data, weights.indices, weights.indptr), shape=weights.shape)
    eucl_squared_ponder = inverse_weights.multiply(eucl_squared).maximum(0).sqrt()

    similarity_matrix[user_ind, :] = eucl_squared_ponder.toarray()
    similarity_matrix[:, user_ind] = eucl_squared_ponder.toarray()


def calc_error(user: int, matrix_ratings: csr_array, mask_ratings: csr_array,
               similarity_matrix: csr_array, metric: int, dist_type: int, k=None) -> float:
    """
    Calculate RMSE/MAE for predicted ratings [these ratings were hidden] for 'user' (30% of all games that 'user' has rated)
    TODO: method 2 to evaluate -> hide randomly in a matrix.

    Parameters
    ----------
        user: int
            User index in ratings
        matrix_ratings : sparse matrix
            User-game ratings matrix
        mask_ratings : sparse matrix
            Mask for user-game ratings matrix
        similarity_matrix : sparse matrix [for EUCLIDEAN], np.ndarray [for COS]
            User-user distance matrix
        metric : "rmse" or "mae" 
            Type of error to calculate
        dist_type : "euclidean" or "cos" 
            Distance type of 'similarity matrix'
    Returns
    -------
        float : MAE / RMSE. !!!np.nan is returned if we couldn't predict any hidden rating
    """

    if (not k):
        k = int(np.sqrt(matrix_ratings.shape[0]))

    # Conversion for efficient modification
    R = matrix_ratings.tolil()  # R : ratings (matrix_ratings)
    M = mask_ratings.tolil()    # M : mask (mask_ratings)

    if isinstance(similarity_matrix, np.ndarray):
        S = similarity_matrix   # S : similarity (similarity_matrx)
    else:
        S = similarity_matrix.tolil()

    error = np.nan  # = nan if cannot be predicted

    # hide games, create new user_game_matrix
    # & memorise to restore later
    old_user_ratings = matrix_ratings[user].copy()
    old_user_mask_ratings = mask_ratings[user].toarray()

    # hide games, modify lil_ratings, lil_mask_ratings in place
    hidden_games = hide_ratings(R, M, user)
    # modify dok_similarity for this row

    if isinstance(similarity_matrix, np.ndarray):
        old_similarity_row = similarity_matrix[user].copy()  # column is not needed since S is symmetric
    else:
        old_similarity_row = similarity_matrix[user].toarray()

    # Recacl similarity_matrix (S) ONLY for USER row, column
    match dist_type:
        case "euclidean":
            recalc_eucl_similarity(user, R, M, S)
        case "cos":
            recalc_cos_similarity(user, R, S)

    # find similar users
    similar_users = get_KNN(S, k=k, user_ind=user)

    # predict ratings (if possible)
    all_ratings, predicted_ratings = predict_ratings_baseline(R, M, similar_users)

    # calc rmse
    to_eval = np.intersect1d(predicted_ratings, hidden_games)

    if to_eval.size != 0:
        match metric:
            case "rmse":
                error = root_mean_squared_error(matrix_ratings[user, to_eval].toarray(), all_ratings[to_eval])
            case "mae":
                error = mean_absolute_error(matrix_ratings[user, to_eval].toarray(), all_ratings[to_eval])

    # restore rows
    R[user] = old_user_ratings
    M[user] = old_user_mask_ratings

    # restore row, col
    S[user, :] = old_similarity_row
    S[:, user] = old_similarity_row  # column = row cause symmetry

    # check if everything was restored

    # if isinstance(similarity_matrix, np.ndarray):
    #     assert ((similarity_matrix == S).all())
    # else:
    #     assert ((similarity_matrix.toarray() == S.toarray()).all())

    # assert ((matrix_ratings.toarray() == R.toarray()).all())
    # assert ((mask_ratings.toarray() == M.toarray()).all())

    return error

# calculate MAE


# def calc_MAE_cos(user: int, matrix_ratings: csr_array, mask_ratings: csr_array, similarity_matrix: np.ndarray) -> float:
#     """Calculate MAE for 'user'.
#     To calculate MAE:
#         - hide 30% of games that 'user' rated
#         - recalc similarity matrix (only rows, cols where user (=vector) participates)
#         - find similar users for 'user'
#         - predict ratings for games (try to predict for all, but if not possible, i.e. no similar user has rated this game,
#             then 0 is given as a rating)
#         - mae is calculated only on these hidden 30% of games & ratings also could be predicted.

#     Parameters
#     ----------
#         user : int
#             User index in matrices
#         matrix_ratings: sparse matrix
#             User-game matrix.
#         mask_ratings : sparse matrix
#             Mask matrix for matrix_ratings to indicate 1 for true rating, 0 for non existent.
#         similarity_matrix : np.ndarray
#             Similarity matrix between users based on cosine distance.
#     Returns
#     -------
#         float : MAE calculated for 'user'.
#     """

#     # Conversion for efficient modification
#     dok_ratings = matrix_ratings.tolil()
#     dok_mask_ratings = mask_ratings.tolil()

#     if not isinstance(similarity_matrix, np.ndarray):
#         dok_similarity = similarity_matrix.todok()
#     else:
#         dok_similarity = similarity_matrix

#     rmse = np.nan  # = nan if cannot be predicted
#     # hide games, create new user_game_matrix
#     # & memorise to restore later
#     old_user_ratings = matrix_ratings[user].toarray()
#     old_user_mask_ratings = mask_ratings[user].toarray()
#     # modify dok ratings, dok_mask_ratings
#     hidden_games = hide_ratings(dok_ratings, dok_mask_ratings, user)
#     # modify dok_similarity for this row
#     old_similarity_row = similarity_matrix[user].copy()
#     recalc_cos_similarity(dok_ratings, dok_similarity, user)
#     # find similar users
#     similar_users = get_KNN(dok_similarity, k=int(np.sqrt(matrix_ratings.shape[0])), user_ind=user)
#     # predict ratings (if possible)
#     all_ratings, predicted_ratings = predict_ratings_baseline(dok_ratings, dok_mask_ratings, similar_users)

#     # calc rmse
#     to_eval = np.intersect1d(predicted_ratings, hidden_games)

#     if to_eval.size != 0:
#         mae = mean_absolute_error(matrix_ratings[user, to_eval].toarray(), all_ratings[to_eval])

#     # Update dok_ratings and dok_mask_ratings in one operation
#     dok_ratings[user] = old_user_ratings
#     dok_mask_ratings[user] = old_user_mask_ratings

#     # Update dok_similarity for the whole row and column in one operation
#     dok_similarity[user, :] = old_similarity_row
#     dok_similarity[:, user] = old_similarity_row

#     # check if everything was restored
#     # assert ((similarity_matrix == dok_similarity).all())
#     # assert ((matrix_ratings.toarray() == dok_ratings.toarray()).all())
#     # assert ((mask_ratings.toarray() == dok_mask_ratings.toarray()).all())

#     return mae
