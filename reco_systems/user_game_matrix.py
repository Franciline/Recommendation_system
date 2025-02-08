import numpy as np
import pandas as pd
from scipy.sparse import csr_array, dok_array, coo_array


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


def get_matrix_user_game(df_reviews: pd.DataFrame):
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
        np.array : Array of ratings means for each user

    """

    # Matrix : row = user, column = game. Missing values (user didnt evaluate the game) are filled with nan
    # df_center, means = center_score(df_reviews)

    matrix_ratings = pd.pivot_table(df_reviews[["User id", "Game id", "Rating"]], values="Rating", dropna=False,
                                    index="User id", columns="Game id", fill_value=np.nan)

    games_table_assoc = pd.Series(data=matrix_ratings.columns)
    users_table_assoc = pd.Series(data=matrix_ratings.index)
    matrix_ratings = matrix_ratings.to_numpy()

    # means = df_reviews[["User id", "Rating"]].groupby("User id", as_index=True).mean()
    # Create matrix with 0 for missing values and 1 for existing values

    nonnans = np.where(~np.isnan(matrix_ratings))
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
    dok_matrix_ratings = csr_array((matrix_ratings[non_zeros], non_zeros), matrix_ratings.shape)
    dok_mask_ratings = csr_array((np.ones(shape=nonnans[0].shape[0]), nonnans), matrix_ratings.shape)
    return dok_matrix_ratings, dok_mask_ratings, users_table_assoc, games_table_assoc
