from collections import Counter

import numpy as np
import pandas as pd
from nltk import bigrams
from rouge import Rouge
from sacrebleu import BLEU

from boardgames_recsys.evaluation.bigram_filters import (
    f_all_comment,
    f_all_comment_unig,
)
from boardgames_recsys.evaluation.bigram_neighbors import _knn_sim, _knn_sim_neg_pos

# -------------------------------------------------

""" Recall (ROUGE like), intersection/nb bigrams user, for each game, no set (with clipping)"""


# using tf idf filtering, count number of intersection in neg and pos
def _intersection_ROUGE_v(
    sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx
):  # user id
    def one_game_score(user_com, sim_users_com):  # NO SET
        # for one game, list of score for all comment with user
        # user_com one row

        user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
        document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens)  # neighbors comment filtered

        # clipping
        df_user_big = pd.DataFrame(Counter(user_big).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_document = pd.DataFrame(Counter(document).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )

        if topx:
            df_document = df_document.head(topx)

        intersection = df_document.merge(df_user_big, on="Bigrams", suffixes=("_neigh", "_user"))
        intersection["Freq_inter"] = intersection[["Freq_neigh", "Freq_user"]].min(axis=1)

        return np.sum(intersection["Freq_inter"]) / len(user_big) if len(user_big) else 0

    # ---------------------

    neg, pos = [], []

    neg_game = user_neg["Game id"].unique()
    pos_game = user_pos["Game id"].unique()

    for game_id in neg_game:
        com_user = user_neg[user_neg["Game id"] == game_id]
        coms_neigh = sim_users_neg[sim_users_neg["Game id"] == game_id]

        neg.append(0) if coms_neigh.empty else neg.append(one_game_score(com_user, coms_neigh))

    for game_id in pos_game:
        com_user = user_pos[user_pos["Game id"] == game_id]
        coms_neigh = sim_users_pos[sim_users_pos["Game id"] == game_id]

        pos.append(0) if coms_neigh.empty else pos.append(one_game_score(com_user, coms_neigh))

    return np.mean(pos) if pos else 0, np.mean(neg) if neg else 0


# type: random, simi, less_simi
def knn_comments_ROUGE_v(
    user_id,
    games_to_consider,
    matrix_ratings,
    mask_ratings,
    cos_sim_matrix,
    users_table,
    games_table,
    comments_all,
    users_mean,
    vectors,
    bigrams_ens,
    type="simi",
    threshold=0,
    k=40,
    topx=None,
):
    """Calculate recall using user comments as reference and neighbor comments as prediction."""

    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos(
        user_id,
        games_to_consider,
        matrix_ratings,
        mask_ratings,
        cos_sim_matrix,
        users_table,
        games_table,
        comments_all,
        users_mean,
        type,
        k,
    )
    pos_prop, neg_prop = _intersection_ROUGE_v(
        sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx
    )
    return pos_prop, neg_prop


# -------------------------------------------------

""" Precision (BLEU like), intersection/nb bigrams neighbors, for each game no set"""


# using tf idf filtering, count number of intersection in neg and pos
def _calc_intersection_BLEU_v(
    sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx
):  # user id
    # calculate the intersection between user and neighbors bigrams

    def one_game_score(user_com, sim_users_com):  # NO SET
        # for one game, list of score for all comment with user
        user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
        document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens)  # neighbors comment filtered

        # clipping
        df_user_big = pd.DataFrame(Counter(user_big).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_document = pd.DataFrame(Counter(document).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )

        if topx:
            df_document = df_document.head(topx)

        intersection = df_document.merge(df_user_big, on="Bigrams", suffixes=("_neigh", "_user"))
        intersection["Freq_inter"] = intersection[["Freq_neigh", "Freq_user"]].min(axis=1)

        return np.sum(intersection["Freq_inter"]) / len(document) if len(document) else 0

    # ---------------------

    neg, pos = [], []

    neg_game = user_neg["Game id"].unique()
    pos_game = user_pos["Game id"].unique()

    for game_id in neg_game:
        com_user = user_neg[user_neg["Game id"] == game_id]
        coms_neigh = sim_users_neg[sim_users_neg["Game id"] == game_id]

        neg.append(0) if coms_neigh.empty else neg.append(one_game_score(com_user, coms_neigh))

    for game_id in pos_game:
        com_user = user_pos[user_pos["Game id"] == game_id]
        coms_neigh = sim_users_pos[sim_users_pos["Game id"] == game_id]

        pos.append(0) if coms_neigh.empty else pos.append(one_game_score(com_user, coms_neigh))

    return np.mean(pos) if pos else 0, np.mean(neg) if neg else 0


# type: random, simi, less_simi
def knn_comments_BLEU_v(
    user_id,
    games_to_consider,
    matrix_ratings,
    mask_ratings,
    cos_sim_matrix,
    users_table,
    games_table,
    comments_all,
    users_mean,
    vectors,
    bigrams_ens,
    type="simi",
    threshold=0,
    k=40,
    topx=None,
):
    """Calculate precision using user comments as reference and neighbor comments as prediction."""

    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos(
        user_id,
        games_to_consider,
        matrix_ratings,
        mask_ratings,
        cos_sim_matrix,
        users_table,
        games_table,
        comments_all,
        users_mean,
        type,
        k,
    )

    score_pos, score_neg = _calc_intersection_BLEU_v(
        sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx
    )
    return score_pos, score_neg


# -----------------------------------------------
# Code for final evaluation versions
#


def _intersection_ROUGE(
    user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, k, topx
):

    def one_game_score(user_com, sim_users_com):  # NO , bigrams
        # user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
        document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens)  # neighbors comment filtered

        # clipping
        df_user_big = pd.DataFrame(Counter(user_com).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_document = pd.DataFrame(Counter(document).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_document = df_document.head(topx)
        # print(df_document)

        intersection = df_document.merge(df_user_big, on="Bigrams", suffixes=("_neigh", "_user"))
        intersection["Freq_inter"] = intersection[["Freq_neigh", "Freq_user"]].min(axis=1)

        return np.sum(intersection["Freq_inter"]) / np.sum(df_user_big["Freq"]) * 100

    def inter(neig, user):
        df_user = pd.DataFrame(Counter(user.split()).items(), columns=["Uni", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_neig = pd.DataFrame(Counter(neig.split()).items(), columns=["Uni", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_neig = df_neig.head(topx)
        # print("taille", len(df_neig))

        intersection = df_neig.merge(df_user, on="Uni", suffixes=("_neigh", "_user"))
        intersection["Freq_inter"] = intersection[["Freq_neigh", "Freq_user"]].min(axis=1)

        # print("intersection!!!")
        # print(intersection['Uni'].unique())

    # ---------------------
    if len(well_predicted_games) == 0:
        return None, None

    score_simi = np.array([0.0, 0.0, 0.0])
    score_rouge1 = np.array([0.0, 0.0, 0.0])
    r = Rouge()

    for game_id in well_predicted_games:
        # print(game_id)
        # users having rated the games
        users_rated = comments_all[comments_all["Game id"] == game_id]["User id"].values
        index_uid = np.where(users_rated == user_id)[0][0]  # first occurrence
        users_rated = np.delete(users_rated, index_uid)

        user_real = comments_all[(comments_all["User id"] == user_id) & (comments_all["Game id"] == game_id)][
            "Comment"
        ].values[0]
        user_com = bigrams(user_real.split())
        user_com = [" ".join(x) for x in user_com]

        # - similare
        m_users = np.intersect1d(knn_all_user[:k], users_rated)
        m = len(m_users)

        neigb_coms = comments_all[(comments_all["User id"].isin(m_users)) & (comments_all["Game id"] == game_id)]
        score_rouge1[0] += r.get_scores(" ".join(neigb_coms["Comment"].values), user_real)[0]["rouge-1"]["r"] * 100
        score_simi[0] += one_game_score(user_com, neigb_coms)
        # print("inter simi")
        # inter(" ".join(neigb_coms['Comment'].values),user_real)

        # - random
        m_random = np.random.choice(users_rated, m, replace=False)
        neigb_coms = comments_all[(comments_all["User id"].isin(m_random)) & (comments_all["Game id"] == game_id)]
        score_rouge1[1] += r.get_scores(" ".join(neigb_coms["Comment"].values), user_real)[0]["rouge-1"]["r"] * 100
        # print("user rand")
        # inter(" ".join(neigb_coms['Comment'].values),user_real)
        score_simi[1] += one_game_score(user_com, neigb_coms)

        # - less similar
        mask = np.isin(knn_all_user, users_rated)
        m_far = knn_all_user[mask][-m:]
        neigb_coms = comments_all[(comments_all["User id"].isin(m_far)) & (comments_all["Game id"] == game_id)]
        score_rouge1[2] += r.get_scores(" ".join(neigb_coms["Comment"].values), user_real)[0]["rouge-1"]["r"] * 100
        # print("user dist")
        # inter(" ".join(neigb_coms['Comment'].values),user_real)
        score_simi[2] += one_game_score(user_com, neigb_coms)

    return score_simi / len(well_predicted_games), score_rouge1 / len(well_predicted_games)


# type: random, simi, less_simi
def knn_ROUGE(
    user_id,
    matrix_ratings,
    mask_ratings,
    cos_sim_matrix,
    users_table,
    games_table,
    comments_all,
    vectors,
    bigrams_ens,
    threshold=0,
    k=40,
    topx=None,
):
    well_predicted_games, knn_all_user = _knn_sim(
        user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, k
    )

    score, score_rouge = _intersection_ROUGE(
        user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, k, topx
    )
    return score, score_rouge


# ---


# for annexe, return score for each game
def _intersection_ROUGE_annexe(
    user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, topx, k
):

    def one_game_score(user_com, sim_users_com, v=False, verb=False):  # NO , bigrams
        # user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)

        document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens)  # neighbors comment filtered

        # clipping
        df_user_big = pd.DataFrame(Counter(user_com).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_document = pd.DataFrame(Counter(document).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_document = df_document.head(topx)

        intersection = df_document.merge(df_user_big, on="Bigrams", suffixes=("_neigh", "_user"))
        intersection["Freq_inter"] = intersection[["Freq_neigh", "Freq_user"]].min(axis=1)

        return (
            np.sum(intersection["Freq_inter"]) / np.sum(df_user_big["Freq"]) * 100
        )  # len(user_com) if len(user_com) else 0

    # ---------------------

    if len(well_predicted_games) == 0:
        return [[], [], []], [[], [], []]

    score_simi = [[], [], []]  # one user, all score for games with different type of users
    score_rouge1 = [[], [], []]
    r = Rouge()

    for game_id in well_predicted_games:
        # print(game_id)

        # users having rated the games
        users_rated = comments_all[comments_all["Game id"] == game_id]["User id"].values
        index_uid = np.where(users_rated == user_id)[0][0]  # first occurrence
        users_rated = np.delete(users_rated, index_uid)

        user_real = comments_all[(comments_all["User id"] == user_id) & (comments_all["Game id"] == game_id)][
            "Comment"
        ].values[0]

        user_com = bigrams(
            comments_all[(comments_all["User id"] == user_id) & (comments_all["Game id"] == game_id)]["Comment"]
            .values[0]
            .split()
        )
        user_com = [" ".join(x) for x in user_com]

        # - similare
        m_users = np.intersect1d(knn_all_user[:k], users_rated)
        m = len(m_users)
        neigb_coms = comments_all[(comments_all["User id"].isin(m_users)) & (comments_all["Game id"] == game_id)]
        score_simi[0].append(one_game_score(user_com, neigb_coms))
        score_rouge1[0].append(r.get_scores(" ".join(neigb_coms["Comment"].values), user_real)[0]["rouge-1"]["r"] * 100)

        # - random
        m_random = np.random.choice(users_rated, min(len(users_rated), m), replace=False)
        neigb_coms = comments_all[(comments_all["User id"].isin(m_random)) & (comments_all["Game id"] == game_id)]
        score_simi[1].append(one_game_score(user_com, neigb_coms))
        score_rouge1[1].append(r.get_scores(" ".join(neigb_coms["Comment"].values), user_real)[0]["rouge-1"]["r"] * 100)

        # - less similar
        mask = np.isin(knn_all_user, users_rated)
        m_far = knn_all_user[mask][-m:]
        neigb_coms = comments_all[(comments_all["User id"].isin(m_far)) & (comments_all["Game id"] == game_id)]
        score_simi[2].append(one_game_score(user_com, neigb_coms))
        score_rouge1[2].append(r.get_scores(" ".join(neigb_coms["Comment"].values), user_real)[0]["rouge-1"]["r"] * 100)

    return score_simi, score_rouge1


# type: random, simi, less_simi
def knn_ROUGE_annexe(
    user_id,
    matrix_ratings,
    mask_ratings,
    cos_sim_matrix,
    users_table,
    games_table,
    comments_all,
    vectors,
    bigrams_ens,
    threshold=0,
    k=40,
    topx=None,
):
    well_predicted_games, knn_all_user = _knn_sim(
        user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, k
    )
    liste_score, liste_rouge1 = _intersection_ROUGE_annexe(
        user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, topx, k
    )

    return np.array(liste_score), np.array(liste_rouge1)


# with rouge1 tf idf and top x
# on all coms


def _intersection_ROUGE12(
    user_id,
    well_predicted_games,
    comments_all,
    knn_all_user,
    threshold,
    vectors_big,
    bigrams_ens,
    vector_unig,
    unig_ens,
    k,
    topx,
):

    def one_game_score(user_com_big, sim_users_com):  # NO, bigrams and unigrams
        # user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
        document = f_all_comment(sim_users_com, vectors_big, threshold, bigrams_ens)  # neighbors comment filtered

        # clipping
        df_user_big = pd.DataFrame(Counter(user_com_big).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_document = pd.DataFrame(Counter(document).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_document = df_document.head(topx)

        intersection = df_document.merge(df_user_big, on="Bigrams", suffixes=("_neigh", "_user"))
        intersection["Freq_inter"] = intersection[["Freq_neigh", "Freq_user"]].min(axis=1)

        predicted = df_document["Bigrams"].unique()
        return np.sum(intersection["Freq_inter"]) / np.sum(df_user_big["Freq"]), predicted

    # ---------------------
    if len(well_predicted_games) == 0:
        return None, None

    score_simi = np.array([0.0, 0.0, 0.0])
    score_rouge1 = np.array([0.0, 0.0, 0.0])
    r = Rouge()
    for game_id in well_predicted_games:
        # users having rated the games
        users_rated = comments_all[comments_all["Game id"] == game_id]["User id"].values
        index_uid = np.where(users_rated == user_id)[0][0]  # first occurrence
        users_rated = np.delete(users_rated, index_uid)

        user_real = comments_all[(comments_all["User id"] == user_id) & (comments_all["Game id"] == game_id)][
            "Lemma"
        ].values[0]
        user_com_unig = user_real.split()
        user_com = bigrams(user_com_unig)
        user_com = [" ".join(x) for x in user_com]

        # - similare
        m_users = np.intersect1d(knn_all_user[:k], users_rated)
        m = len(m_users)
        neigb_coms = comments_all[(comments_all["User id"].isin(m_users)) & (comments_all["Game id"] == game_id)]
        r2 = one_game_score(user_com, neigb_coms)
        score_simi[0] += r2[0] * 100
        # rouge 1 on filtered neighbors
        s = (
            r.get_scores(f_all_comment_unig(neigb_coms, vector_unig, threshold, unig_ens), user_real)[0]["rouge-1"]["r"]
            * 100
        )
        score_rouge1[0] += s

        # - random
        m_random = np.random.choice(users_rated, m, replace=False)
        neigb_coms = comments_all[(comments_all["User id"].isin(m_random)) & (comments_all["Game id"] == game_id)]
        s = (
            r.get_scores(f_all_comment_unig(neigb_coms, vector_unig, threshold, unig_ens), user_real)[0]["rouge-1"]["r"]
            * 100
        )
        score_rouge1[1] += s
        r2 = one_game_score(user_com, neigb_coms)
        score_simi[1] += r2[0] * 100

        # - less similar
        mask = np.isin(knn_all_user, users_rated)
        m_far = knn_all_user[mask][-m:]
        neigb_coms = comments_all[(comments_all["User id"].isin(m_far)) & (comments_all["Game id"] == game_id)]
        s = (
            r.get_scores(f_all_comment_unig(neigb_coms, vector_unig, threshold, unig_ens), user_real)[0]["rouge-1"]["r"]
            * 100
        )
        score_rouge1[2] += s

        r2 = one_game_score(user_com, neigb_coms)
        score_simi[2] += r2[0] * 100

    return score_simi / len(well_predicted_games), score_rouge1 / len(well_predicted_games)


# type: random, simi, less_simi
def knn_ROUGE12(
    user_id,
    matrix_ratings,
    mask_ratings,
    cos_sim_matrix,
    users_table,
    games_table,
    comments_all,
    vectors,
    bigrams_ens,
    vector_unig,
    unig_ens,
    threshold=0,
    k=40,
    topx=None,
):
    well_predicted_games, knn_all_user = _knn_sim(
        user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, k
    )

    score, score_rouge = _intersection_ROUGE12(
        user_id,
        well_predicted_games,
        comments_all,
        knn_all_user,
        threshold,
        vectors,
        bigrams_ens,
        vector_unig,
        unig_ens,
        k,
        topx,
    )
    return score, score_rouge


# FOR BLEU PLOT


def _intersection_ROUGE_prim(
    user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, k, topx
):

    def one_game_score(user_com, sim_users_com, v=False):  # NO , bigrams
        # user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
        document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens)  # neighbors comment filtered

        # clipping
        df_user_big = pd.DataFrame(Counter(user_com).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_document = pd.DataFrame(Counter(document).items(), columns=["Bigrams", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        df_document = df_document.head(topx)
        # print(df_document)

        intersection = df_document.merge(df_user_big, on="Bigrams", suffixes=("_neigh", "_user"))
        intersection["Freq_inter"] = intersection[["Freq_neigh", "Freq_user"]].min(axis=1)

        if v:
            print("doc")
            print(df_document["Bigrams"].unique())
            print(user_com)
        return np.sum(intersection["Freq_inter"]) / np.sum(df_user_big["Freq"])

    def mean_bleu(bleu, neigb_coms, user_com):
        # compute mean bleu score for users and neigh
        score = 0
        for hyp in neigb_coms:
            score += bleu.sentence_score(hypothesis=hyp, references=[user_com]).score
        return score / len(neigb_coms) if len(neigb_coms) else 0

    # ---------------------
    if len(well_predicted_games) == 0:
        return None, None, None

    score_simi = np.array([0.0, 0.0, 0.0])
    score_rouge1 = np.array([0.0, 0.0, 0.0])
    score_bleu = np.array([0.0, 0.0, 0.0])
    v = False
    r = Rouge()
    bleu = BLEU(max_ngram_order=2, effective_order=True)

    if user_id == 1193 or user_id == 1903:
        v = True

    for game_id in well_predicted_games:
        if v:
            print(user_id, game_id)
        # users having rated the games
        users_rated = comments_all[comments_all["Game id"] == game_id]["User id"].values
        index_uid = np.where(users_rated == user_id)[0][0]  # first occurrence
        users_rated = np.delete(users_rated, index_uid)

        user_real = comments_all[(comments_all["User id"] == user_id) & (comments_all["Game id"] == game_id)][
            "Lemma"
        ].values[0]
        user_com = bigrams(user_real.split())
        user_com = [" ".join(x) for x in user_com]

        # - similare
        m_users = np.intersect1d(knn_all_user[:k], users_rated)
        m = len(m_users)

        neigb_coms = comments_all[(comments_all["User id"].isin(m_users)) & (comments_all["Game id"] == game_id)]
        hyp = " ".join(neigb_coms["Lemma"].values)
        score_rouge1[0] += r.get_scores(hyp, user_real)[0]["rouge-1"]["r"] * 100
        # score_bleu[0] += mean_bleu(bleu, neigb_coms['Lemma'].values, user_real)
        score_bleu[0] += bleu.sentence_score(hyp, [user_real]).score
        score_simi[0] += one_game_score(user_com, neigb_coms) * 100

        # - random
        m_random = np.random.choice(users_rated, m, replace=False)
        neigb_coms = comments_all[(comments_all["User id"].isin(m_random)) & (comments_all["Game id"] == game_id)]
        hyp = " ".join(neigb_coms["Lemma"].values)
        score_rouge1[1] += r.get_scores(hyp, user_real)[0]["rouge-1"]["r"] * 100
        # score_bleu[1] += mean_bleu(bleu, neigb_coms['Lemma'].values, user_real)
        score_bleu[1] += bleu.sentence_score(hyp, [user_real]).score
        score_simi[1] += one_game_score(user_com, neigb_coms) * 100

        # - less similar
        mask = np.isin(knn_all_user, users_rated)
        m_far = knn_all_user[mask][-m:]
        neigb_coms = comments_all[(comments_all["User id"].isin(m_far)) & (comments_all["Game id"] == game_id)]
        hyp = " ".join(neigb_coms["Lemma"].values)
        score_rouge1[2] += r.get_scores(hyp, user_real)[0]["rouge-1"]["r"] * 100
        # score_bleu[2] += mean_bleu(bleu, neigb_coms['Lemma'].values, user_real)
        score_bleu[2] += bleu.sentence_score(hyp, [user_real]).score
        score_simi[2] += one_game_score(user_com, neigb_coms) * 100

    return (
        score_simi / len(well_predicted_games),
        score_rouge1 / len(well_predicted_games),
        score_bleu / len(well_predicted_games),
    )


# type: random, simi, less_simi
def knn_ROUGE_prim(
    user_id,
    matrix_ratings,
    mask_ratings,
    cos_sim_matrix,
    users_table,
    games_table,
    comments_all,
    vectors,
    bigrams_ens,
    threshold=0,
    k=40,
    topx=None,
):
    # no threshold no top x!! bleu score
    well_predicted_games, knn_all_user = _knn_sim(
        user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, k
    )

    score, score_rouge, score_bleu = _intersection_ROUGE_prim(
        user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, k, topx
    )
    return score, score_rouge, score_bleu
