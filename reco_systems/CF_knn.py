import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, nan_euclidean_distances
from scipy.sparse import csr_array

# Comment style
"""

Parameters
----------
Returns
-------
"""


def calc_similarity_matrix(matrix_ratings, mask_matrix, dist_type: str):
    """Calculate similarity matrix (users). Similarity is based on a certain type of distance (e.g. euclidean, cosine).

    EUCLIDEAN : 
        If user X and user Y has rated nothing in common, then the value of the euclidean distance is nan
        which is then replaced by 0 (to use csr_array), hence the mask matrix is needed (for prediction).
    Parameters
    ----------
        matrix_ratings: csr_array:
            Matrix of ratings with row = user, column = game
        dist_type: str
            Type of distance which would be used as a metric for similarity between users.
    Returns
    -------
        if dist_type = "cos", then 
            np.ndarray, None : Similarity matrix
        if dist_type = "euclidean", then
            csr_array, csr_array : Similarity matrix, mask for similarity matrix 
    """

    similarity_matrix = None
    match dist_type:
        case "cos":
            similarity_matrix = cosine_distances(matrix_ratings.tocsr())
        case "euclidean":
            similarity_matrix = _eucl_sparse(matrix_ratings, mask_matrix)
        # case "manhattan":
        #     similarity_matrix = manhattan_distances(matrix_ratings)
        case _:
            pass
    return similarity_matrix


def _eucl_sparse(matrix_ratings, mask_ratings):
    R, M = matrix_ratings, mask_ratings  # just for abbreviation

    # Eucl_distance(R, R) = R^2 + R^2 - 2dot(R * R.T)
    # 1. Calc R^2 and apply mask
    R_2 = R.multiply(R)
    R_2_masked = R_2.dot(M.T)  # by applying mask only remaining values would be those on common rated games

    # No need to apply mask to dot product 'cause 0 values are not counted
    # negative values are replaced with 0
    eucl_squared = (R_2_masked + R_2_masked.T - 2 * R.dot(R.transpose())).maximum(0)

    # Calc ponderation. Every eucl_dist(u1, u2) is divided by common ratings shared by u1, u2
    weights = M.dot(M.transpose())
    inverse_weights = csr_array((1/weights.data, weights.indices, weights.indptr), shape=weights.shape)

    # .sqrt is not necessary
    return eucl_squared.multiply(inverse_weights).sqrt()


def get_KNN(similarity_matrix: np.ndarray, k: int, user_ind: int, dtype: str, mask_sim_matrix=None) -> np.array:
    """
    Find k nearest neighbors (similarity = distance wise) for a given user (user_ind).

    Parameters
    ----------
        similarity_matrix: np.ndarray
            Similarity (between users) matrix.
        k: int

        user_ind:int
            User's (for whom we provide recommendations) index (=id) in similarity matrix.

    Returns
    -------
        np.array: Array of row indices of k nearest users.
    """
    # row corresponding to the user
    if not isinstance(similarity_matrix, np.ndarray):
        sim_user_row = similarity_matrix[user_ind].toarray()
    else:
        sim_user_row = similarity_matrix[user_ind].copy()

    if dtype == "euclidean":
        # values corresponding to no intersection between rated games
        to_replace = ~mask_sim_matrix[user_ind].toarray().astype(bool)
        sim_user_row[to_replace] = np.inf
    # argpartition in O(n)
    sim_user_row[user_ind] = np.inf  # to prevent choosing user himself
    ksmallest = np.argpartition(sim_user_row, kth=min(k, sim_user_row.shape[0]-1))
    # print(similarity_matrix[user_ind][ksmallest])
    return ksmallest[:k]


def weight_avg_distance(similarity_matrix: csr_array, similar_users: np.array, matrix_ratings: csr_array, user_ind: int, means) -> np.array:
    """
    Calculate ponderated average (scaled back) of games ratings by users distances to each other.The weight W of a distance d = 1/d.

    Parameters
    ----------
        similarity_matrix:
            Matrix User-User where every value is a distance.
        similar_users:
            Users (IDs) similar to 'user_ind'
        matrix_ratings:
            User-game matrix
        user_ind: User for who games ratings should be predicted
    Returns
    -------
        Predicted ratings for games. If no similar user rated the game X, then the predicted rating is X.
    """

    distances = similarity_matrix[user_ind, similar_users]

    # Inversing distances
    non_zeros = distances.nonzero()
    distances[non_zeros] = np.reciprocal(distances[non_zeros])

    # Ratings which are non zero
    mask = matrix_ratings[similar_users].transpose().toarray() != 0

    prediction = np.dot(distances, matrix_ratings[similar_users].toarray())  # numerator
    sums = np.array([np.sum(distances[mask_row]) for mask_row in mask])  # denominator

    nonzeros = sums.nonzero()  # non zero values in denominator

    # calc final weighted average
    prediction[nonzeros] = prediction[nonzeros] / sums[nonzeros] + means[user_ind]
    return prediction


def weight_avg_nb_reviews(df_reviews: pd.DataFrame, similar_users: np.array, matrix_ratings: csr_array, user_ind: int, means) -> np.array:
    """
    Calculate ponderated average (scaled back) of games ratings by number of reviews.

    Parameters
    ----------
        df_reviews:
            jeux_clean.csv
        similar_users:
            Users (IDs) similar to 'user_ind'
        matrix_ratings:
            User-game matrix
        user_ind: User for who games ratings should be predicted
    Returns
    -------
        Predicted ratings for games. If no similar user rated the game X, then the predicted rating is X.
    """

    users_nb_reviews = df_reviews[["User id", "Game id"]].groupby("User id", as_index=True).count()
    weights = users_nb_reviews.loc[similar_users]["Game id"].values
    # Ratings which are non zero
    mask = matrix_ratings[similar_users].transpose().toarray() != 0
    prediction = np.dot(weights, matrix_ratings[similar_users].toarray())  # numerator

    sums = np.array([np.sum(weights[mask_row]) for mask_row in mask])  # denominator

    nonzeros = sums.nonzero()  # non zero values in denominator

    # calc final weighted average
    prediction[nonzeros] = prediction[nonzeros] / sums[nonzeros] + means[user_ind]
    return prediction


# def predict_ratings(weighted_means: np.ndarray, coefs: np.array = None) -> np.array:
#     """Combine several predicted ratings (weighted averages) into one.

#     Parameters
#     ----------
#         weighted_means
#             Weighted average for ratings (already scaled back).
#         coefs
#             Coefficients (weights of each ponderation). If None, means are considered as equal.
#     """

#     if coefs is None:
#         return np.sum(weighted_means, axis=0) / weighted_means.shape[0]
#     m = weighted_means.shape[1]
#     return np.sum(np.tile(coefs, (m, 1)) * weighted_means.T, axis=1)


def predict_ratings_baseline(matrix_ratings, mask_matrix, similar_users: np.array) -> np.array:
    """
    Baseline. Predict ratings for all existing games in 'matrix_ratings'. If the rating cannot be predicted (i.e. there is no similar user
    who has rated the game), then the rating is 0.

    Parameters
    ----------
        matrix_ratings: csr_array
            Matrix user-game with given ratings (0 if missing value or rating = 0)
        mask_matrix: csr_array
            Mask matrix (mask_matrix.shape = matrix_ratings.shape) with 1 if rating exists, 0 if not.
        simialar_users: np.array
            Array of users indices in 'matrix_ratings'. Ratings will be predicted based only on their ratings.
    Returns
    -------
        np.array : Array of predicted ratings for each game
        np.array : Games indices for which ratings could be predicted, i.e. if no similar user has rated the game, then 
            its index wouldn't be included
    """

    users_ratings = matrix_ratings[similar_users]             # ratings of similar users
    valid_count = mask_matrix[similar_users].sum(axis=0)      # number of existing rating (for division to calc mean)
    games_means = np.zeros(shape=(users_ratings.shape[1], ))  # put 0 for ratings where no ratings are known
    # calc means
    return np.divide(users_ratings.sum(axis=0), valid_count, out=games_means, where=valid_count != 0), np.nonzero(valid_count)[0]


def get_games_ind(predicted_ratings: np.array, n: int) -> np.array:
    """
    Parameters
    ----------
    Returns
        similar_users: np.array


        n: int
            Top n games will be chosen for recommendation.
    -------
        np.array : Array of selected games indices (columns in a matrix)
    """

    return np.argpartition(-predicted_ratings,  kth=min(n, predicted_ratings.size))[:n]


def get_games_df(df_games: pd.DataFrame, table_assoc: pd.Series, selected_games: np.array) -> pd.DataFrame:
    """
    Select rows in a dataframe
    Parameters
    ----------
        df_games: DataFrame
            jeux_clean.csv
        table_assoc: Series (.index = indices used, .values = true IDs)
            Table of association between games ids (in DB) and their indices in matrix, arrays etc.
        selected_games: np.array
            Array of indices of chosen for recommendation games.
    Returns
    -------
        Part of 'df_games' with lines associated to selected for recommendation games.
    """

    return df_games[df_games["Game id"].isin(table_assoc[selected_games])]


# Version finale : tout faire dans la fonction + save graphique ?
def distance_evolution(matrix_ratings, mask_matrix, k: int, user_ind: int) -> np.array:
    """
    Graphie évolution distance pour un user
    """
    # Get similarity matrix (get_sim_matrix ?)

    similarity_matrix, _ = calc_similarity_matrix(matrix_ratings, mask_matrix, dist_type="euclidean")

    # Faire knn, extraire dist
    voisins = get_KNN(similarity_matrix, k, user_ind)
    get_dists = np.vectorize(lambda x: similarity_matrix[user_ind][x])
    distances = get_dists(voisins)

    x_data = np.arange(int(max(distances)+1), step=1)
    nb_nn = np.vectorize(lambda x: (distances[distances < x]).size)
    y_data = nb_nn(x_data)

    #
    plt.scatter(x_data, y_data)
    plt.xlabel = "k"
    plt.ylabel = "Nombre de users dont la distance à " + user_ind + " est inférieure à k"
    plt.show()
    return None
