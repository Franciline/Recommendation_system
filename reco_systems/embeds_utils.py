from sacrebleu import corpus_bleu
from math import ceil
from reco_systems.lemmatization import *
from reco_systems.evaluation import *
from reco_systems.CF_knn import *
from reco_systems.text_filtering import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy
import pandas as pd
import numpy as np
import seaborn as sns
from sacrebleu import BLEU

ALLOW_ERR = 2  # max error of ratings prediction for comments generation


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


def get_sim_users(user_id, matrix_ratings, mask_ratings, cos_dist_matrix, users_table, games_table, k, specific_game):

    MR, mask = matrix_ratings.tolil(), mask_ratings.tolil()
    user_ind = users_table[users_table["User id"] == user_id]["User index"].item()

    # prev_ratings, prev_mask_ratings = MR[user_ind, :], mask[user_ind, :],
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

    # save changes
    new_cos_row = cos_dist_matrix[user_ind, :]

    # restore
    # MR[user_ind, :], mask[user_ind, :] = prev_ratings, prev_mask_ratings
    cos_dist_matrix[user_ind, :], cos_dist_matrix[:, user_ind] = prev_sim, prev_sim

    diff = np.abs(matrix_ratings[user_ind, hidden_games] - pred_ratings[hidden_games])

    return user_ind, diff, sim_users, hidden_games, pred_ratings, new_cos_row


def get_sim_per_game(game_ind, sim_users, mask_ratings):
    """Get a number of similar users that has rated the game"""

    return int(mask_ratings[sim_users, game_ind].sum())


def choose_users(target_user_ind, sim_users, n, users_type, game_ind, game_id, mask_ratings, dist_to_users, comments_clusters, users_table):
    """Choose n users that has rated the game based on type:
        - "similar"
        - "random"
        - "distant"
    """
    users = mask_ratings[:, game_ind].nonzero()[0]
    users = users[users != target_user_ind]

    if users_type == "similar":  # in this case, number of similar users who has rated the game should be exactly n
        select_users = np.intersect1d(sim_users, users)
        return users_table[users_table["User index"].isin(select_users)]["User id"].values

    select_users = []
    if users_type == "random":
        select_users = np.random.choice(users, size=n, replace=False)
        users_ids = users_table[users_table["User index"].isin(select_users)]["User id"].values

        # Resample while not finding user that was not deleted
        while (comments_clusters[(comments_clusters["Game id"] == game_id) & (comments_clusters["User id"].isin(users_ids))].empty):
            select_users = np.random.choice(users, size=n, replace=False)
            users_ids = users_table[users_table["User index"].isin(select_users)]["User id"].values

    if users_type == "distant":
        distances = dist_to_users[users]
        select_users = users[np.argpartition(-distances, kth=n-1)[:n]]
        users_ids = users_table[users_table["User index"].isin(select_users)]["User id"].values

    return users_ids


def eval_all_embeddings(user_id, matrix_ratings, mask_ratings, users_table, games_table,
                        cos_dist_matrix, k, comments_clusters, clusters_weights,
                        comments_lemmatized, lemmas, weight_type="count", specific_game=None):

    user_ind, diff, sim_users, hidden_games, pred_ratings, dist_to_users = get_sim_users(user_id, matrix_ratings,
                                                                                         mask_ratings, cos_dist_matrix,
                                                                                         users_table, games_table, k, specific_game)
    correct_ratings = hidden_games[np.argwhere(diff < ALLOW_ERR).flatten()]
    # games_ids = games_table[games_table["Game index"].isin(correct_ratings)]["Game id"].values

    comments_clusters, clusters_weights = hide_comments(user_id, games_table, hidden_games,
                                                        comments_clusters, clusters_weights, weight_type)

    all_scores, phrases = [], []
    for game_ind in correct_ratings:
        nb_users = get_sim_per_game(game_ind, sim_users, mask_ratings)
        user_scores = []
        no_append = False

        for users_type in ["similar", "random", "distant"]:
            game_id = games_table[games_table["Game index"] == game_ind]["Game id"].item()
            users_ids = choose_users(user_ind, sim_users, nb_users, users_type, game_ind, game_id, mask_ratings,
                                     dist_to_users, comments_clusters, users_table)

            rouge1, rouge2, bleu, pred_phrases = find_phrases(user_id, game_id, users_ids,
                                                              comments_clusters, clusters_weights, comments_lemmatized,
                                                              lemmas, weight_type=weight_type)
            if (np.nan in [rouge1, rouge2, bleu]):
                # print([rouge1, rouge2, bleu], users_type, nb_users, users_ids, game_id)
                no_append = True
                break
            if specific_game:
                phrases.append(pred_phrases)
            user_scores.append([user_ind, rouge1, rouge2, bleu, game_id])

        if no_append:  # no phrases found (happens when phrases is in deleted cluster)
            continue

        all_scores.append(user_scores)
    if specific_game:
        return all_scores, phrases
    return all_scores


def find_phrases(target_user_id, game_id, other_users_id,
                 comments_clusters, clusters_weights, comments_lemmatized, lemmas,
                 return_phrases=False, weight_type="count"):

    other_users_comments = comments_clusters[comments_clusters["User id"].isin(other_users_id)
                                             & (comments_clusters["Game id"] == game_id)]

    if (other_users_comments.empty):
        # print("EMPTY", target_user_id, other_users_id, game_id)
        return np.nan, np.nan, np.nan

    phrases, phrases_per_cluster = 15, 1
    sim_clusters = other_users_comments["Cluster"].unique()
    n_clusters, n_clusters_goal = len(sim_clusters), phrases // phrases_per_cluster

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

        other_users_comments = other_users_comments[other_users_comments["Cluster"].isin(cw)]
    else:
        phrases_per_cluster = ceil(phrases / n_clusters)  # -> several phrases per cluster

    # In each cluster find mean embedding
    cluster_means = other_users_comments.groupby("Cluster")["Embedding"].mean().reset_index()
    phrases = []

    cw = clusters_weights.set_index("Cluster")
    for cluster, mean in cluster_means.itertuples(index=False):
        mean = mean.reshape(1, -1)

        cluster_embeds = comments_clusters[comments_clusters["Cluster"] == cluster]

        distances = euclidean_distances(np.array(cluster_embeds["Embedding"].tolist()), mean).flatten()
        kth = min(distances.size, phrases_per_cluster)
        dist_sorted = np.argpartition(distances, kth-1)[:kth]
        sel_phrases = cluster_embeds.iloc[dist_sorted]["Phrases"]
        phrases.extend(sel_phrases)

    phrases = pd.DataFrame(data={"Phrases": phrases})
    phrases = phrases.assign(Length=phrases["Phrases"].str.split().apply(len))

    if return_phrases:
        return phrases

    phrases_lemma = lemmatize_comment(
        phrases, "/Users/bsh2022/Study/L3/Projet_recherche/TreeTagger", lemmas).reset_index()

    real_comment = comments_lemmatized[(comments_lemmatized["Game id"] == game_id) & (
        comments_lemmatized["User id"] == target_user_id)]["Lemma"].item()

    rouge1, rouge2 = rouge_ngram_scores(" ".join(phrases_lemma["Lemma"]), real_comment)
    bleu = bleu_ngram_scores(phrases_lemma["Lemma"].str.cat(sep=" "), real_comment)
    return rouge1, rouge2, bleu, phrases


def rouge_ngram_scores(pred_phrases, real_comment):
    def get_ngram_overlap(p, r, n):
        vector = CountVectorizer(ngram_range=(n, n))
        vec = vector.fit_transform([p, r]).toarray()
        pred_vec, ref_vec = vec[0], vec[1]

        overlap = np.sum(np.minimum(pred_vec, ref_vec))
        recall = overlap / np.sum(ref_vec)
        return recall * 100

    rouge1_recall = get_ngram_overlap(pred_phrases, real_comment, 1)
    rouge2_recall = get_ngram_overlap(pred_phrases, real_comment, 2)

    return rouge1_recall, rouge2_recall


def bleu_ngram_scores(pred_phrases, real_comment):
    bleu = BLEU(lowercase=True, force=False, max_ngram_order=2, effective_order=True)
    return bleu.corpus_score(pred_phrases, [real_comment]).score


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
