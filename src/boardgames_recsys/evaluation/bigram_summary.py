"""Summary and plotting helpers for bigram evaluation notebooks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def words_freq(data, corpus) -> pd.DataFrame:
    """
    Construction d'un dataframe avec la fréquence des mots dans un corpus
    """

    lem, occurences = np.unique(data["Lemma"], return_counts=True)

    df = pd.DataFrame({"Lemma": lem, "Freq": occurences})
    nb_comments = data["Comment line"].nunique()
    df["Freq"] = df["Freq"].apply(lambda val: val / nb_comments)

    # Garder uniquement les lemmas qui appraissent dans le corpus
    # return df[df["Lemma"].isin(corpus)]
    return df


def construction_corpus(lemmas: pd.DataFrame, taille: int) -> dict:
    """Construction d'un corpus à partir d'une BDD de commentaires."""

    lemmas = lemmas[~lemmas["Lemma"].isna()]
    lemmas = lemmas[lemmas["Part of speech"].isin(["ADJ", "NOM", "VER", "NEG"])]
    lemmas = lemmas[
        ~lemmas["Lemma"].isin(
            [
                "bref",
                "bof",
                "excelent",
                "bon",
                "autre",
                "seul",
                "tendre",
                "fin"
                "super",
                "superbe",
                "juste",
                "jouable",
                "ca",
                "faire",
                "pouvoir",
                "ausi",
            ]
        )
    ]
    lemmas = lemmas["Lemma"].to_numpy()

    lem, occ = np.unique(lemmas, return_counts=True)
    freq_lem = pd.DataFrame({"lemma": lem, "freq": occ})

    freq_lem = freq_lem.sort_values(by=["freq"], ascending=False)
    return freq_lem.head(taille)["lemma"].to_numpy()


def type_user_count_df(simi_pos, random_pos, less_simi_pos, simi_neg, random_neg, less_simi_neg):
    """Return dataframe of positive/negative review counts by neighbor type."""

    simi = pd.DataFrame({"count": simi_pos, "User": "Similar", "Type": "Positive Reviews"})
    random = pd.DataFrame({"count": random_pos, "User": "Random", "Type": "Positive Reviews"})
    less_simi = pd.DataFrame({"count": less_simi_pos, "User": "Less Similar", "Type": "Positive Reviews"})

    df_posneg = pd.concat([simi, less_simi, random])

    simi = pd.DataFrame({"count": simi_neg, "User": "Similar", "Type": "Negative Reviews"})
    random = pd.DataFrame({"count": random_neg, "User": "Random", "Type": "Negative Reviews"})
    less_simi = pd.DataFrame({"count": less_simi_neg, "User": "Less Similar", "Type": "Negative Reviews"})

    return pd.concat([df_posneg, simi, less_simi, random])


def df_user_type_mean(df_posneg):
    """Return dataframe with the mean of each category."""

    group_means = df_posneg.groupby(["User", "Type"])["count"].mean().reset_index()
    group_means["Type"] = group_means["Type"].replace(
        {"Negative Reviews": "Mean Negative Reviews", "Positive Reviews": "Mean Positive Reviews"}
    )
    user_order = ["Similar", "Random", "Less Similar"]
    group_means["User"] = pd.Categorical(group_means["User"], categories=user_order, ordered=True)
    return group_means


def plot_posnegviolin(data, means, title="", xlabel="", ylabel="", figname="", save=False):
    """Plot evaluation results for neighbor type, sentiment, and category mean."""

    plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=data,
        x="User",
        y="count",
        hue="Type",
        density_norm="width",
        order=["Similar", "Random", "Less Similar"],
        cut=0,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    custom_palette = {
        "Mean Positive Reviews": "#4554ff",
        "Mean Negative Reviews": "#ff6f00",
    }

    if means is not None:
        sns.stripplot(
            data=means,
            x="User",
            y="count",
            hue="Type",
            dodge=True,
            hue_order=["Mean Positive Reviews", "Mean Negative Reviews"],
            jitter=True,
            marker="o",
            palette=custom_palette,
        )
    if save:
        plt.savefig(f"../images/{figname}.png")


def lst_avg(liste, nb_iters, n_users):
    """Return per-user mean over `nb_iters` for each list in `liste`."""

    return [np.mean(np.array(sublist).reshape(nb_iters, n_users), axis=0) for sublist in liste]


def evaluate_big(
    func,
    users,
    nb_iters,
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
    threshold=0.13,
    k=40,
    topx=None,
):
    """Evaluate a bigram function for three neighbor selection strategies."""

    n_users = len(users)

    avg_pos_s = []
    avg_neg_s = []
    avg_pos_ls = []
    avg_neg_ls = []
    avg_pos_r = []
    avg_neg_r = []

    for _ in range(nb_iters):
        for user_id in users:
            pos, neg = func(
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
                k=k,
                threshold=threshold,
                topx=topx,
            )
            avg_pos_s.append(pos)
            avg_neg_s.append(neg)

            pos, neg = func(
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
                type="less_simi",
                k=k,
                threshold=threshold,
            )
            avg_pos_ls.append(pos)
            avg_neg_ls.append(neg)

            pos, neg = func(
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
                type="random",
                k=k,
                threshold=threshold,
            )
            avg_pos_r.append(pos)
            avg_neg_r.append(neg)

    lst_l = lst_avg([avg_pos_s, avg_pos_r, avg_pos_ls, avg_neg_s, avg_neg_r, avg_neg_ls], nb_iters, n_users)

    df_pos_neg = type_user_count_df(*lst_l)
    group_means = df_user_type_mean(df_pos_neg)

    return df_pos_neg, group_means
