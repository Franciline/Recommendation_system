import numpy as np
import pandas as pd
from scipy.sparse import csr_array, dok_array, coo_array


def center_score(df: pd.DataFrame):
    """
    Center the ratings, with xi the score of user i, xi = xi - avg(i) 
    Necessary columns: 'User id' 'Rating'. Used to avoid biais in scores.

    Parameters
    ----------
        df: dataframe, expected avis_clean 
    
    Returns 
    -------
        Two pd.DataFrames:
         - Copy of the df with the ratings centered
         - A df with the average rate of each users, cols = ["User id", "Average rate"]
    """

    df = df.copy(deep=True)

    # centering the scores
    mean_score = df[["User id", "Rating"]].groupby("User id").mean().rename(columns={"Rating": "Average rate"})
    mean_score = df.merge(mean_score, on="User id")
    mean_score['Rating'] -=  mean_score['Average rate']
    
    return mean_score.drop(columns=['Average rate']), mean_score[["User id", "Average rate"]].drop_duplicates()


def normalize(df: pd.DataFrame):
    """
    Normalize the ratings, with xi the score of user i, xi = (xi - min(i)) / (max(i) - min(i)) 
    Necessary columns: 'User id' 'Rating'.

    Parameters
    ----------
        df: dataframe, expected avis_clean 
    
    Returns 
    -------
        Two pd.DataFrames:
         - Copy of the df with the ratings normalized
         - A df with the min and max rate of each users, cols = ["User id", "Max", "Min"]
    """

    df = df.copy(deep=True)
    min_max = df.groupby("User id").agg({"Rating": ['min', 'max']}).reset_index()
    min_max.columns = ["User id", "Min", "Max"]  # no index levels
    min_max = df.merge(min_max, on="User id")
    
    min_max['Rating'] = (min_max['Rating'] - min_max["Min"])/(min_max["Max"] - min_max["Min"])
    min_max['Rating'] = min_max['Rating'].fillna(0) # NaN to 0 from division by 0

    return min_max.drop(columns=["Min", "Max"]), min_max[["User id", "Max", "Min"]].drop_duplicates()


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
        csr_array: User-game sparse matrix
        csr_array: Mask (sparse matrix) for user-game matrix. 1 : existing rating, 0 : missing value.
        pd.Series : table which associate user index in matrices with his true id in DB.
        pd.Series : table which associate game index in matrices with its true id in DB.

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
