import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from scipy.sparse import csr_array

# Comment style
"""

Parameters
----------
Returns
-------
"""


def center_score(df: pd.DataFrame):
    """
    avis_clean
    The df with at least 3 columns 'User id' 'Game id' 'Rating', center the scores:
    with xi the score of user i, xi = xi - mean(i)

    To avoid biais between scores
    Returns the df with the scores centered
    """
    df = df.copy(deep=True)

    # df with average rate of each user
    mean_score = df[["User id", "Rating"]].groupby("User id").mean().rename(
        columns={"Rating": "Average rate"})
    mean_score = df.merge(mean_score, on="User id")
    mean_score['Rating'] -= mean_score['Average rate']

    df['Rating'] = mean_score['Rating']
    return df, mean_score[["User id", "Average rate"]].drop_duplicates()


def normalize(df: pd.DataFrame):
    """
    avis_clean
    The df with at least 2 columns, 'User id', 'Rating', normalize the scores:
    score xi of user i: xi = (xi - min)/(max - min) with min and max the corresponding values
    for the user i
    Returns the df with the scores normalized
    """

    df = df.copy(deep=True)
    min_max = df.groupby("User id").agg({"Rating": ['min', 'max']}).reset_index()
    min_max.columns = ["User id", "Min", "Max"]  # no index levels
    min_max = df.merge(min_max, on="User id")

    df['Rating'] = (min_max['Rating'] - min_max["Min"])/(min_max["Max"] - min_max["Min"])
    # df['Rating'] = df['Rating'].fillna(0) # NaN from division by 0

    return df, min_max[["User id", "Max", "Min"]].drop_duplicates()


def get_matrix_user_game(df_reviews: pd.DataFrame) -> tuple[csr_array, np.array]:
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
        np.array : Array of ratings means for each user.
    """

    # Matrix : row = user, column = game. Missing values (user didnt evaluate the game) are filled with nan
    # df_center, means = center_score(df_reviews)

    matrix_ratings = pd.pivot_table(df_reviews[["User id", "Game id", "Rating"]], values="Rating", dropna=False,
                                    index="User id", columns="Game id", fill_value=np.nan).to_numpy()

    # means = df_reviews[["User id", "Rating"]].groupby("User id", as_index=True).mean()
    # Create matrix with 0 for missing values and 1 for existing values

    mask_matrix = np.where(np.isnan(matrix_ratings), 0, 1)
    non_zeros_mask = mask_matrix.nonzero()
    np.nan_to_num(matrix_ratings, copy=False, nan=0.0)

    # drop games where all ratings are nan
    # matrix_ratings = df.dropna(axis=1, how='all').to_numpy()
    # users_means = np.nanmean(matrix_ratings, axis=1)[:, None]  # means by rows
    # matrix_ratings = matrix_ratings - users_means  # remove biais
    # np.nan_to_num(matrix_ratings, copy=False, nan=0.0)  # replace nan values with 0.0
    # Sparse matrix (since lots of 0). Optimized on row slicing
    # csr_array((data, (row_ind, col_ind)), [shape=(M, N)])
    # rows, cols = matrix_ratings.nonzero()  # non zero values in matrix_ratings, two lines only if we save manhattan
    # non_zeros = rows.astype(np.int32), cols.astype(np.int32)
    non_zeros = matrix_ratings.nonzero()
    return csr_array((matrix_ratings[non_zeros], non_zeros), matrix_ratings.shape), \
        csr_array((mask_matrix[non_zeros_mask], non_zeros_mask), mask_matrix.shape)


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
            similarity_matrix = cosine_distances(matrix_ratings)
        case "euclidean":
            similarity_matrix = euclidean_distances(matrix_ratings)
        case "manhattan":
            similarity_matrix = manhattan_distances(matrix_ratings)
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
    prev_value = similarity_matrix[user_ind][user_ind]
    similarity_matrix[user_ind][user_ind] = np.inf  # to prevent choosing user himself
    ksmallest = np.argpartition(similarity_matrix[user_ind], kth=min(k, similarity_matrix.shape[1]))
    # print(similarity_matrix[user_ind][ksmallest])
    similarity_matrix[user_ind][user_ind] = prev_value
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


def predict_ratings_baseline(matrix_ratings: csr_array, mask_matrix: csr_array, similar_users: np.array) -> np.array:
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
        np.array : Array of predicted ratings for each game.
    """

    users_ratings = matrix_ratings[similar_users]             # ratings of similar users
    valid_count = mask_matrix[similar_users].sum(axis=0)      # number of existing rating (for division to calc mean)

    games_means = np.zeros(shape=(users_ratings.shape[1], ))  # put 0 for ratings where no ratings are known
    return np.divide(users_ratings.sum(axis=0), valid_count, out=games_means, where=valid_count != 0)  # calc means


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
    table_assoc_games = games_reviews.loc[games_reviews["User id"] > min_reviews].reset_index()["Game id"]

    df_filtered = df_reviews.copy()
    df_filtered.loc[~df_filtered["Game id"].isin(table_assoc_games.values), ["Game id", "Rating"]] = np.nan

    return df_filtered, table_assoc_games


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
