import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_array

# Comment style
"""

Parameters
----------
Returns
-------
"""


def get_matrix_user_game(df_reviews: pd.DataFrame) -> csr_array:
    """
    Create matrix where row = user, column = game. Value in row I and column J is the rating that user I gave to the game J.
    Missing values (if user didn't evaluate the game) are filled with zeros.

    Parameters
    ----------
        df_reviews: DataFrame
            avis_clean.csv

    Returns
    -------
        csr_array: Sparse matrix (optimized on row slicing)
    """

    # Matrix : row = user, column = game. Missing values (user didnt evaluate the game) are filled with zeros
    matrix_ratings = pd.pivot_table(df_reviews[["User id", "Game id", "Rating"]],
                                    values="Rating", index="User id", columns="Game id", aggfunc="mean", fill_value=0).to_numpy()

    non_zeros = matrix_ratings.nonzero()  # non zero values in matrix_ratings
    # Sparse matrix (since lots of 0). Optimized on row slicing
    # csr_array((data, (row_ind, col_ind)), [shape=(M, N)])
    return csr_array((matrix_ratings[non_zeros], non_zeros), matrix_ratings.shape)


def calc_similarity_matrix(matrix_ratings, dist_type: str) -> np.ndarray:
    """Calculate similarity matrix (users). Similarity is based on a certain type of distance (e.g. euclidean, cosine).

    Parameters
    ----------
        matrix_ratings: csr_array or? np.ndarray):
            Matrix of ratings with row = user, column = game
        dist_type: str
            Type of distance which would be used as a metric for similarity between users.
    Returns
    -------
        np.ndarray: Similarity matrix
    """

    similarity_matrix = None  # ? np.empty(shape = matrix_ratings.shape)
    match dist_type:
        case "cos":
            # Users similarity (pairwise)
            # 1 - to evaluate on min distance on KNN
            similarity_matrix = 1 - cosine_similarity(matrix_ratings)
        case "euclidean":
            pass
        case _:
            pass
    return similarity_matrix


def get_KNN(similarity_matrix: np.ndarray, k: int, user_ind: int) -> np.array:
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

    # argpartition in O(n)
    ksmallest = np.argpartition(similarity_matrix[user_ind], kth=min(k, similarity_matrix.shape[1]))
    return ksmallest[:k]


def get_games_ind(matrix_ratings: np.ndarray, similar_users: np.array, n: int) -> np.array:
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

    # TO DO
    # for NOW only : take simple mean to choose games (NO distance ponderation, no users mean rescaling)
    games_means = matrix_ratings[similar_users].mean(axis=0)  # axis=0 -> along columns
    return np.argpartition(-games_means, kth=min(n, games_means.size))[:n]


def filter_games(df_reviews: pd.DataFrame, min_reviews: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create a 'table' of associattion between games ids in database and corresponding indices for matrices, arrays.
    Parameters
    ----------
        df_reviews: pd.DataFrame
            avis_clean.csv
        min_reviews: int
            Min number of reviews that a game should have to be considered by the system.
    Returns
    -------
        pd.DataFrame : df_reviews with some games excluded
        pd.Series : Association between Game_id (in DB) and its index which will be used in matrices, arrays etc.
            (.index = indices used, .values = true IDs)
    """

    games_reviews = df_reviews[["Game id", "User id"]].groupby("Game id", as_index=True).count()
    table_assoc = games_reviews[games_reviews["User id"] > min_reviews].reset_index()["Game id"]
    return df_reviews[df_reviews["Game id"].isin(table_assoc.values)], table_assoc


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
