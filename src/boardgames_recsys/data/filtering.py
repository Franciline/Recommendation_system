import pandas as pd


def filter_df(df_reviews: pd.DataFrame, min_reviews: int, max_iter: int = 10) -> pd.DataFrame:
    """
    Filter users & games so that in the returned dataframe each user gave at least minimum number of reviews, the same for games.

    Parameters
    ----------
        df_reviews:
            avis_clean.csv
        min_reviews:int
        max_iter:int
    Returns
    -------
        pd.DataFrame : DataFrame where each user has given at least 'min reviews' reviews, the same for game (if convergenve is reached
        beofre 'max iter' iterations). Otherwise, return dataframe resulting from 'max iter' iterations.
    """

    if (max_iter == 0):
        return df_reviews

    # Print remaining number of users and remaining number of games
    print(df_reviews["User id"].nunique(), df_reviews["Game id"].nunique())

    # Goal : eliminate users who gave less than < min_reviews
    # Dataframe produced : index, User id, Number of rated games per user [this column is named as Game id]
    count_users = df_reviews[["Game id", "User id"]].groupby("User id").count().reset_index()

    # Reduce DataFrame 'df_reviews' which will contain only users who gave at least 'min reviews'
    reduced = df_reviews[df_reviews["User id"].isin(
        count_users[count_users["Game id"] >= min_reviews]["User id"])]  # delete users

    # Goal : eliminate games who gave less than < min_reviews (users are already deleted)
    # Dataframe produced : index, Game id, Number of users who rated the game [this columns is named as User id]
    count_games = reduced.groupby("Game id").count().reset_index()

    # Reduce DataFrame 'reduced' which will contain only users who gave at least 'min reviews' AND
    # games who received at least 'min reviews' from these users
    reduced = reduced[reduced["Game id"].isin(
        count_games[count_games["User id"] >= min_reviews]["Game id"])]  # delete games

    # Check convergence
    if (df_reviews.equals(reduced) or len(reduced) == 0):
        return reduced

    # If not : reiterate (recursion) till convergence
    else:
        return filter_df(reduced, min_reviews, max_iter - 1)
