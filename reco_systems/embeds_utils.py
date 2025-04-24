from math import ceil
from reco_systems.lemmatization import *
from reco_systems.evaluation import *
from reco_systems.CF_knn import *
from reco_systems.text_filtering import *

from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy
import pandas as pd
import numpy as np
import seaborn as sns


def cluster_weight_entropy(comments_clusters):
    table = comments_clusters.groupby(["Cluster", "User id"]).size().unstack(fill_value=0)
    table = table.div(table.sum(axis=1), axis=0)
    return pd.DataFrame(data={"Cluster": table.index, "Weight": entropy(table.to_numpy(), axis=1)})


def cluster_weight_count(comments_clusters):
    """Calculates weight based on a number of reviews."""
    weight = comments_clusters.groupby("Cluster")["Phrases"].count().reset_index().rename(columns={"Phrases": "Weight"})
    return weight


def hide_comments(target_user_id, games_table, hidden_games_ind, comments_clusters, clusters_weights, weight_type="count"):
    """Hide comments (phrases) given by target user id on hidden games. Also updates weights."""
    games_ids = games_table[games_table["Game index"].isin(hidden_games_ind)]["Game id"].values

    cc, cw = comments_clusters, clusters_weights.set_index("Cluster")
    mask_drop = (cc["User id"] == target_user_id) & (cc["Game id"].isin(games_ids))

    if weight_type == "count":
        weight_user = (cc.loc[mask_drop].groupby("Cluster", sort=False)["Phrases"].size())
        cw.loc[:, "Weight"] = cw["Weight"].sub(weight_user, fill_value=0)

    if weight_type == "entropy":
        clusters = comments_clusters.loc[mask_drop, "Cluster"]
        new_weights = cluster_weight_entropy(
            comments_clusters[comments_clusters["Cluster"].isin(clusters)]).set_index("Cluster")
        cw.update({"Weight": new_weights["Weight"]})
    return cc.loc[~mask_drop].sort_values("Cluster", ignore_index=True), cw.reset_index()


def eval_embeddings(user_id, matrix_ratings, mask_ratings, users_table, games_table,
                    cos_dist_matrix, k, users_means, pos_comments_clusters, neg_comments_clusters,
                    pos_clusters_weights, neg_clusters_weights, pos_centroids, neg_centroids,
                    comments_lemmatized, lemmas, specific_game=None, weight_type="count"):

    MR, mask = matrix_ratings.tolil(), mask_ratings.tolil()

    user_ind = users_table[users_table["User id"] == user_id]["User index"].item()

    prev_ratings, prev_mask_ratings = MR[user_ind, :], mask[user_ind, :],
    prev_sim = cos_dist_matrix[user_ind, :]

    if specific_game is None:
        hidden_games = hide_ratings(MR, mask, user_ind, 0.1)
    else:
        specific_game_ind = games_table[games_table["Game id"] == specific_game]["Game index"].item()
        hidden_games = np.array([specific_game_ind])
        MR[user_ind, specific_game_ind], mask[user_ind, specific_game_ind] = 0, 0

    recalc_cos_similarity(user_ind, MR, cos_dist_matrix)

    # choice of similar users
    sim_users = get_KNN(cos_dist_matrix, k, user_ind)

    pred_ratings, mask_pred_ratings = predict_ratings_baseline(MR, mask, sim_users, cos_dist_matrix, user_ind)
    # consider only games that the system was able to predict
    hidden_games = np.intersect1d(hidden_games, mask_pred_ratings)

    # restore
    MR[user_ind, :], mask[user_ind, :] = prev_ratings, prev_mask_ratings
    cos_dist_matrix[user_ind, :], cos_dist_matrix[:, user_ind] = prev_sim, prev_sim

    diff = np.abs(matrix_ratings[user_ind, hidden_games] - pred_ratings[hidden_games])

    ALLOW_ERR = 3

    user_mean = users_means.loc[users_means["User id"] == user_id, "Rating"].item()
    pos, neg = pred_ratings[hidden_games] < user_mean, pred_ratings[hidden_games] > user_mean

    correct_neg = hidden_games[np.argwhere(neg & (diff < ALLOW_ERR)).flatten()]
    correct_pos = hidden_games[np.argwhere(pos & (diff < ALLOW_ERR)).flatten()]

    # Find games ids
    pos_game_ids = games_table[games_table["Game index"].isin(correct_pos)]["Game id"].values
    neg_game_ids = games_table[games_table["Game index"].isin(correct_neg)]["Game id"].values

    # Find users ids
    sim_users_ids = users_table[users_table["User index"].isin(sim_users)]["User id"].values

    # Treat positive ratings
    print(
        f"Correct predicted ratings neg : {neg_game_ids.size}, pos : {pos_game_ids.size} ({hidden_games.size} hidden)")

    if specific_game is not None:
        if correct_neg.size > 0:
            return find_phrases(user_id, specific_game, sim_users_ids,
                                *hide_comments(user_id, games_table, hidden_games,
                                               neg_comments_clusters, neg_clusters_weights, weight_type),
                                neg_centroids, comments_lemmatized, lemmas, return_phrases=True, weight_type=weight_type)
        return find_phrases(user_id, specific_game, sim_users_ids,
                            *hide_comments(user_id, games_table, hidden_games,
                                           pos_comments_clusters, pos_clusters_weights, weight_type),
                            pos_centroids, comments_lemmatized, lemmas, return_phrases=True, weight_type=weight_type)

    bigrams, unigrams = [], []
    for game in pos_game_ids:
        b, u = find_phrases(user_id, game, sim_users_ids,
                            *hide_comments(user_id, games_table, hidden_games,
                                           pos_comments_clusters, pos_clusters_weights),
                            pos_centroids, comments_lemmatized, lemmas, weight_type=weight_type)
        bigrams.append(b)
        unigrams.append(u)

    # Treat negative ratings
    for game in neg_game_ids:
        b, u = find_phrases(user_id, game, sim_users_ids,
                            *hide_comments(user_id, games_table, hidden_games,
                                           neg_comments_clusters, neg_clusters_weights),
                            neg_centroids, comments_lemmatized, lemmas, weight_type=weight_type)
        bigrams.append(b)
        unigrams.append(u)

    return bigrams, unigrams


def find_phrases(target_user_id, game_id, sim_users_ids,
                 comments_clusters, clusters_weights, centroids, comments_lemmatized, lemmas,
                 return_phrases=False, weight_type="count"):

    sim_users_comments = comments_clusters[comments_clusters["User id"].isin(
        sim_users_ids) & (comments_clusters["Game id"] == game_id)]

    if (sim_users_comments.empty):
        return np.nan, np.nan

    phrases, phrases_per_cluster = 15, 1
    sim_clusters = sim_users_comments["Cluster"].unique()
    print("Init", len(sim_clusters))
    n_clusters, n_clusters_goal = len(sim_clusters), phrases // phrases_per_cluster

    # diff = n_clusters_goal - n_clusters  # number of other clusters to take
    # n_closest = diff // n_clusters

    print("game", game_id, "number of similar users comments", sim_users_comments.size, "n clusters",
          n_clusters, "phrases per cluster", phrases_per_cluster, end=" ")

    if n_clusters > n_clusters_goal:  # clusters selection should be made -> 1 phrase per cluster
        n_clusters = n_clusters_goal
        cw = clusters_weights[clusters_weights["Cluster"].isin(sim_clusters)].sort_values("Weight")
        if n_clusters > 1:
            half = ceil(n_clusters / 2)
            cw = pd.concat([cw.tail(half), cw.head(half)])["Cluster"].values

        else:
            if weight_type == "count":  # take with fewest phrases (in a hope to be more specific)
                cw = pd.concat([cw.head(n_clusters)])["Cluster"].values
            else:  # entropy -> take with highest entropy
                cw = pd.concat([cw.tail(n_clusters)])["Cluster"].values

        sim_users_comments = sim_users_comments[sim_users_comments["Cluster"].isin(cw)]
    else:
        phrases_per_cluster = ceil(phrases / n_clusters)  # -> several phrases per cluster

    # if n_clusters < n_clusters_goal:
    #     other_centroids_df = centroids[~centroids["Cluster"].isin(sim_clusters)]
    #     other_centroids = np.array(other_centroids_df["Embedding"].tolist())

    # In each cluster find mean embedding
    cluster_means = sim_users_comments.groupby("Cluster")["Embedding"].mean().reset_index()
    phrases = []

    cw = clusters_weights.set_index("Cluster")

    for cluster, mean in cluster_means.itertuples(index=False):
        mean = mean.reshape(1, -1)

        cluster_embeds = comments_clusters[comments_clusters["Cluster"] == cluster]

        distances = cosine_distances(np.array(cluster_embeds["Embedding"].tolist()), mean).flatten()
        kth = min(distances.size, phrases_per_cluster)
        dist_sorted = np.argpartition(distances, kth-1)[:kth]
        sel_phrases = cluster_embeds.iloc[dist_sorted]["Phrases"]
        phrases.extend(sel_phrases)

        # if n_clusters < n_clusters_goal and diff > 0:
        #     closest_clusters = np.argpartition(cosine_distances(other_centroids, mean).flatten(), n_closest)[:n_closest]
        #     closest_clusters = other_centroids_df.iloc[closest_clusters]["Cluster"]
        #     for cluster in closest_clusters:
        #         other_phrases = comments_clusters[comments_clusters["Cluster"] == cluster]
        #         phrase_index = np.argmin(cosine_distances(
        #             np.array(other_phrases["Embedding"].tolist()), mean).flatten())
        #         phrases.append(other_phrases.iloc[phrase_index]["Phrases"])
        #         diff -= 1

    phrases = pd.DataFrame(data={"Phrases": phrases})
    phrases = phrases.assign(Length=phrases["Phrases"].str.split().apply(len))

    if return_phrases:
        return phrases

    phrases_lemma = lemmatize_comment(
        phrases, "/Users/bsh2022/Study/L3/Projet_recherche/TreeTagger", lemmas).reset_index()

    real_comment = comments_lemmatized[(comments_lemmatized["Game id"] == game_id) & (
        comments_lemmatized["User id"] == target_user_id)]["Lemma"].item()
    pred_phrases = " ".join(phrases_lemma["Lemma"])

    # Intersecting bigrams
    vector = CountVectorizer(ngram_range=(2, 2))
    bigrams = vector.fit_transform([pred_phrases, real_comment]).toarray()

    # Intersecting unigrams
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    unigrams = vectorizer.fit_transform([pred_phrases, real_comment]).toarray()

    v1 = CountVectorizer(ngram_range=(1, 1))
    v1.fit_transform([pred_phrases])
    print("Unigrams in total", len(v1.get_feature_names_out()))
    # ROUGE score (intersection with counting frequencies)
    uni_inter, bi_inter = np.sum(np.minimum(unigrams[0, :], unigrams[1, :])), np.sum(
        np.minimum(bigrams[0, :], bigrams[1, :]))
    print(bi_inter / np.sum(bigrams[1, :]) * 100, uni_inter / np.sum(unigrams[1, :]) * 100)
    return bi_inter / np.sum(bigrams[1, :]) * 100, uni_inter / np.sum(unigrams[1, :]) * 100


def plot_bi_uni_grams(users, unigrams, bigrams):
    sns.set_theme(rc={"figure.figsize": (15, 6)})

    unigrams_df = pd.DataFrame(data={"User id": users, "Unigrams": unigrams})
    bigrams_df = pd.DataFrame(data={"User id": users, "Bigrams": bigrams})

    unigrams_df = unigrams_df.explode("Unigrams")
    unigrams_df = unigrams_df[~unigrams_df["Unigrams"].isna()]

    bigrams_df = bigrams_df.explode("Bigrams")
    bigrams_df = bigrams_df[~bigrams_df["Bigrams"].isna()]

    ax = sns.stripplot(data=unigrams_df, x="User id", y="Unigrams")
    sns.pointplot(data=unigrams_df, x="User id", y="Unigrams", ax=ax,
                  color="orange", label="Mean unigrams", errorbar=None)
    sns.pointplot(data=bigrams_df, x="User id", y="Bigrams", ax=ax, color="green", label="Mean bigrams", errorbar=None)

    ax.set_ylabel("ROUGE score (in %)")
    ax.set_title("Pos/neg separation 50 top users")
