from scipy.sparse import csr_array
import pandas as pd
import numpy as np
from scipy.sparse import csr_array, lil_array, dok_array
from sklearn.metrics.pairwise import cosine_distances, nan_euclidean_distances
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from .CF_knn import get_KNN, predict_ratings_baseline, calc_similarity_matrix
from typing import Union


from typing import Union

# Evaluation ver 2 : hide randomly x% of ratings everywhere in a matrix


def _hide_ratings_full_matrix(matrix_ratings: dok_array, mask_ratings: dok_array, percentage: float = 0.3) -> tuple[np.array, np.array]:
    """Hide x% of existing ratings anywhere in a matrix.
    Modify INPLACE 'matrix_ratings', 'mask_ratings'

    Returns
    -------
        np.array : users indices whose ratings were hidden
        np.array : games indices whose ratings were hidden
        Those array have the same size and one-to-one correpondence.
        For example, if users = [0, 0, 2], games = [0, 1, 2], then for user 0 ratings for games 0 and 1 were hidden,
        and for user 1 the rating for game 2 was hidden.
    """

    rows, cols = mask_ratings.nonzero()  # existing ratings indices
    indices = np.arange(rows.size)
    indices_to_hide = np.random.choice(indices, int(percentage * indices.size), replace=False)

    # rows_to_hide : users, cols_to_hide : games
    rows_to_hide, cols_to_hide = rows[indices_to_hide], cols[indices_to_hide]

    mask_ratings[rows_to_hide, cols_to_hide] = 0
    matrix_ratings[rows_to_hide, cols_to_hide] = 0

    return rows_to_hide, cols_to_hide


def _treat_user(user_ind: int, all_hidden_users: np.array, all_hidden_games: np.array,
                similarity_matrix: Union[np.ndarray, csr_array], ratings_hidden: dok_array,
                mask_hidden: dok_array, matrix_ratings: csr_array,
                k: int, metric: str) -> tuple[float, int]:
    """Auxiliary function for 'calc_error_full_matrix' to predict ratings for one user.

    Parameters
    ----------
        user_ind: int
            User index in matrices for who ratings should be predicted

        all_hidden_users: np.array
            Indices (rows) of users for whom some ratings were hidden (see '_hide_ratings_full_matrix')
        all_hidden_games: np.array
            Indices (columns) of games for which some ratings were hidden (see '_hide_ratings_full_matrix')

        similarity_matrix: np.ndarray (for 'cos') or sparse matrix (for 'euclidean')
            User-user similarity matrix [recalculated for user-game matrix with hidden games]

        ratings_hidden : dok_array
            User-game matrix but with x% hidden ratings
        mask_hidden : dok_array
            Mask for user-game matrix 'ratings_hidden'

        matrix_ratings : csr_array
            Original user-game matrix [needed to calculate error]

        k: int
            k for KNN algorithm (to find similar users)
        metric: str
            "rmse" or "mae" or "rmse_mae"

    Returns: !!!
        !Intermediate values! to calculate rmse, mae later.
        Intermediate values : sum(true_rating - predicted_rating) (no division) and N : number of ratings that we were able to predict

        -> If metric = "rmse" or "mae", then return sum, np.nan, N
        -> If metric = "rmse_mae", then return sum_for_rmse, sum_for_mae, N
        -> If nothing could be predicted for a given user, then return np.nan, np.nan, 0
    """

    user_hidden_ratings = all_hidden_games[all_hidden_users == user_ind]
    similar_users = get_KNN(similarity_matrix, k=k, user_ind=user_ind)

    all_pred_ratings, able_to_predict = predict_ratings_baseline(ratings_hidden, mask_hidden, similar_users)
    to_eval = np.intersect1d(able_to_predict, user_hidden_ratings)

    if to_eval.size == 0:
        return np.nan, np.nan, 0  # coudn't predict anything

    diff = matrix_ratings[user_ind, to_eval] - all_pred_ratings[to_eval]
    count_predicted = to_eval.size

    match metric:
        case "rmse":
            sum_error = np.dot(diff, diff), np.nan
        case "mae":
            sum_error = np.sum(np.absolute(diff)), np.nan
        case "rmse_mae":
            sum_error = np.dot(diff, diff), np.sum(np.absolute(diff))
    return *sum_error, count_predicted


def calc_error_full_matrix(matrix_ratings: csr_array, mask_ratings: csr_array,
                           metric: str, dist_type: str, k: int = None) -> tuple[float, float, int]:
    """Evaluate quality of rating's prediction by hiding 20% of all existing ratings anywhere in User-game matrix.

    Parameters
    ----------
        matrix_ratings: sparse matrix
            User-game ratings matrix
        mask_ratings: sparse matrix
            Mask for user-game ratings (to know if rating truly exists)
        metric: str
            "rmse" or "mae" or "rmse_mae"
        dist_type: str
            "cos" or "euclidean"
        k: int, default = sqrt(number of users)
            k for KNN algorithm (to find similar users)

    Returns
    -------
        N : number of ratings that couldn't be predicted

        -> If metric = "rmse" or "mae", then return RMSE or MAE, N
        -> If metric = "rmse_mae", then return RMSE, MAE, N
        -> If we weren't able to predict anything for every selected (for who at least 1 rating was hidden) user,
            then if metric = "rmse" or "mae" -> np.nan, N
                 if metric = "rmse_mae" -> np.nan, np.nan, N
    """

    if not k:
        k = int(np.sqrt(matrix_ratings.shape[0]))  # k = sqrt(nb users)

    # conversion for efficient modification of sparsity structure
    ratings_hidden = matrix_ratings.todok()
    mask_hidden = mask_ratings.todok()

    # hide 20% of games
    users_hidden, games_hidden = _hide_ratings_full_matrix(ratings_hidden, mask_hidden, percentage=0.2)
    # recalc similarity matrix
    sim_matrix_hidden = calc_similarity_matrix(ratings_hidden.tocsr(), mask_hidden.tocsr(), dist_type)
    print(f"Number of hidden ratings : {users_hidden.size} ({matrix_ratings.data.size} existing ratings)")
    # hide 20% of his games
    treat_user_vect = np.vectorize(_treat_user, excluded=(1, 2, 3, 4, 5, 6, 7, 8), otypes=['f', 'f', 'i'])

    # sum1 = sum_rmse or sum_mae, sum2 = sum_mae or np.nan
    sum1, sum2, count_predicted = treat_user_vect(np.unique(users_hidden), users_hidden, games_hidden,
                                                  sim_matrix_hidden, ratings_hidden, mask_hidden, matrix_ratings,
                                                  k, metric)

    sum_count = np.sum(count_predicted)
    nb_non_predicted = users_hidden.size - sum_count

    match metric:
        case "rmse" | "mae":
            if sum_count == 0:
                return np.nan, nb_non_predicted
            return np.sqrt(np.nansum(sum1) / sum_count), nb_non_predicted

        case "rmse_mae":
            if sum_count == 0:
                return np.nan, np.nan, nb_non_predicted

            sum_rmse, sum_mae = np.nansum(sum1), np.nansum(sum2)
            return np.sqrt(sum_rmse / sum_count), np.sqrt(sum_mae / sum_count), nb_non_predicted

        case _:
            return np.nan


# Evaluation VERSION 1. Hide x% of ratings for some selected users
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
               similarity_matrix: csr_array, metric: str, dist_type: str, k=None) -> tuple[float, float]:
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
        If metric = "rmse" or "mae" -> return float (rmse or mae), np.nan
        If metric = "rmse_mae" -> return rmse, mae (float, float)
        If nothing could be predicted -> return np.nan, np.nan
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

    error = np.nan, np.nan  # = nan if cannot be predicted

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
                error = root_mean_squared_error(matrix_ratings[user, to_eval].toarray(), all_ratings[to_eval]), np.nan
            case "mae":
                error = mean_absolute_error(matrix_ratings[user, to_eval].toarray(), all_ratings[to_eval]), np.nan
            case "rmse_mae":
                error = root_mean_squared_error(matrix_ratings[user, to_eval].toarray(), all_ratings[to_eval]), \
                    mean_absolute_error(matrix_ratings[user, to_eval].toarray(), all_ratings[to_eval])

    # restore rows
    R[user] = old_user_ratings
    M[user] = old_user_mask_ratings

    # restore row, col
    S[user, :] = old_similarity_row
    S[:, user] = old_similarity_row  # column = row cause symmetry
    if error[0] > 6:
        print(user, to_eval.size)
    # check if everything was restored

    # if isinstance(similarity_matrix, np.ndarray):
    #     assert ((similarity_matrix == S).all())
    # else:
    #     assert ((similarity_matrix.toarray() == S.toarray()).all())

    # assert ((matrix_ratings.toarray() == R.toarray()).all())
    # assert ((mask_ratings.toarray() == M.toarray()).all())

    return error

# RMSE & MAE means


def calc_RMSE_MAE_mean(k: np.ndarray, user_count: pd.DataFrame, min_reviews: int, max_reviews: int, matrix_ratings: csr_array, mask_ratings: csr_array, similarity_matrix: np.ndarray, dist_type: str):
    """
        Calculate the RMSE and MAE for every user matching the number of reviews requierments for using K-NN method for each given K.

        Parameters :
        ----------
            k (np.ndarray) : k for K-nn
            user_count (pd.DataFrame) : DF with User id and Review_count
            min_reviews, max_reviews (int) :
            matrix_ratings :
            mask_ratings :
            similarity_matrix :
            dist_type :

        Returns :
            (filtered_users.size, 4) shaped DF, columns = ['User id', k, 'Type' : RMSE, MAE, 'value']
    """

    # User filtering based on number of reviews
    np.random.seed(1)
    filtered = user_count[(user_count['Count reviews'] >= min_reviews) & (
        user_count['Count reviews'] <= max_reviews)]
    filtered = filtered.sample(min(100, filtered.shape[0]))

    users = filtered[['User index']].to_numpy().flatten()
    print(users.size)
    # Dataframe creation
    data_rmse, data_mae = [], []

    for tmp_k in k:
        np.random.seed(1)  # to evaluate on the same games
        # otypes = tuple[float, float]
        vect_rsme_mae = np.vectorize(calc_error, excluded=(1, 2, 3, 4, 5, 6), otypes=['f', 'f'])

        rmse, mae = vect_rsme_mae(users, matrix_ratings, mask_ratings,
                                  similarity_matrix, metric="rmse_mae",
                                  dist_type=dist_type, k=tmp_k)
        data_rmse.append(rmse)
        data_mae.append(mae)

    # List to Dataframe conversion

    df = pd.DataFrame({"RMSE": data_rmse, "MAE": data_mae, "K": k})
    # expand every list, .reset_index() may be not necessary
    df = df.apply(lambda x: x.explode())

    # divide RMSE into Value and Type, the same for MAE
    df = df.melt(id_vars="K", value_vars=["RMSE", "MAE"], var_name="Type", value_name="Value")

    df['K'] = df['K'].astype(np.int64)
    df['Value'] = df['Value'].astype(np.float64)
    df['Type'] = df['Type'].astype(str)
    return df
