
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

from reco_systems.user_game_matrix import *
from reco_systems.CF_knn import *
from reco_systems.evaluation import *


# Counting freq, for bleu rouge

# for corpus
def words_freq(data, corpus) -> pd.DataFrame:
    """
    Construction d'un dataframe avec la fréquence des mots dans un corpus
    """

    lem, occurences = np.unique(data['Lemma'], return_counts=True)

    df = pd.DataFrame({'Lemma': lem, 'Freq': occurences})
    nb_comments = data["Comment line"].nunique()
    df['Freq'] = df['Freq'].apply(lambda val: val/nb_comments)

    # Garder uniquement les lemmas qui appraissent dans le corpus
    #return df[df['Lemma'].isin(corpus)]
    return df

def construction_corpus(lemmas:pd.DataFrame, taille: int) -> dict:
    """ 
    Construction d'un corpus à partir d'une BDD de commentaires
    avis.colums = 'Comment title', 'Comment body'

    Retourne df avec mots du corpus et leurs fréquences, les 'taille' plus fréquentes
    """

    # Corpus creation from lemmatized dataframe
    lemmas = lemmas[~lemmas["Lemma"].isna()]
    lemmas = lemmas[lemmas['Part of speech'].isin(['ADJ', 'NOM', "VER", "NEG"])]
    lemmas = lemmas[~lemmas["Lemma"].isin(["bref", "bof", "excelent", "bon", "autre", "seul", "tendre", "fin"
                                           "super", "superbe", "juste", "jouable", "ca", "faire", "pouvoir", "ausi"])]
    lemmas = lemmas['Lemma'].to_numpy()

    # Occurencies calculation for each lemma
    lem, occ = np.unique(lemmas, return_counts=True)
    freq_lem = pd.DataFrame({'lemma': lem, 'freq': occ})

    freq_lem = freq_lem.sort_values(by=['freq'], ascending=False)
    return freq_lem.head(taille)['lemma'].to_numpy()

# functions for evaluation

def type_user_count_df(simi_pos, random_pos, less_simi_pos, simi_neg, random_neg, less_simi_neg):
    simi = pd.DataFrame({"count" : simi_pos, "User" : "Similar","Type" : "Positive Reviews"})
    random = pd.DataFrame({"count" : random_pos, "User" : "Random","Type" : "Positive Reviews"})
    less_simi = pd.DataFrame({"count" : less_simi_pos, "User" : "Less Similar","Type" : "Positive Reviews"})

    df_posneg = pd.concat([simi, less_simi, random])

    simi = pd.DataFrame({"count" : simi_neg, "User" : "Similar","Type" : "Negative Reviews"})
    random = pd.DataFrame({"count" : random_neg, "User" : "Random","Type" : "Negative Reviews"})
    less_simi = pd.DataFrame({"count" : less_simi_neg, "User" : "Less Similar","Type" : "Negative Reviews"})

    return pd.concat([df_posneg, simi, less_simi, random])

def df_user_type_mean(df_posneg):
    group_means = df_posneg.groupby(['User', 'Type'])['count'].mean().reset_index()
    group_means['Type'] = group_means['Type'].replace({'Negative Reviews': 'Mean Negative Reviews','Positive Reviews': 'Mean Positive Reviews'})
    user_order = ['Similar', 'Random', 'Less Similar'] 
    group_means['User'] = pd.Categorical(group_means['User'], categories=user_order, ordered=True)
    return group_means

def plot_posnegviolin(data, means, title='', xlabel='', ylabel='', figname='', save = False):
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=data, x="User", y="count", hue="Type",density_norm='width',order=["Similar", "Random", "Less Similar"], cut=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    custom_palette = {
        'Mean Positive Reviews': '#4554ff', # nuance blue  
        'Mean Negative Reviews': '#ff6f00', # nuance orange
    }
    if means is not None:
        sns.stripplot(
            data=means,
            x='User',
            y='count',
            hue='Type',
            dodge=True,
            hue_order=['Mean Positive Reviews', 'Mean Negative Reviews'],       
            jitter=True,     
            marker='o', 
            palette=custom_palette,       
        )
    if save:
        plt.savefig(f"../images/{figname}.png")

def lst_avg(liste, nb_iters, n_users):
    # ! order
    return [np.mean(np.array(sublist).reshape(nb_iters, n_users), axis = 0) for sublist in liste]

def evaluate_big(func, users, nb_iters, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, threshold=0.13, k=40, topx=None):
    # users_ids_samp = comments_all.sample(n=n_users)['User id']
    n_users = len(users)
    
    avg_pos_s = []
    avg_neg_s = []
    avg_pos_ls = []
    avg_neg_ls = []
    avg_pos_r = []
    avg_neg_r = []

    for _ in range(nb_iters):
        for id in users:
            # simi
            pos, neg = func(id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type='simi', k=40, threshold=threshold, topx = topx)
            avg_pos_s.append(pos)
            avg_neg_s.append(neg)

            # less_simi
            pos, neg = func(id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type='less_simi', k=40, threshold=threshold)
            avg_pos_ls.append(pos)
            avg_neg_ls.append(neg)

            # random
            pos, neg = func(id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type='random', k=40, threshold=threshold)
            avg_pos_r.append(pos)
            avg_neg_r.append(neg)
        
    lst_l = lst_avg([avg_pos_s, avg_pos_r, avg_pos_ls, avg_neg_s, avg_neg_r, avg_neg_ls], nb_iters, n_users)

    df_pos_neg = type_user_count_df(*lst_l)
    group_means = df_user_type_mean(df_pos_neg)

    return df_pos_neg, group_means


# functions for the intersection

def create_df(ngram_finder, ngram_stat):
        bigram_freq = ngram_finder.score_ngrams(ngram_stat)

        bigrams_df = pd.DataFrame(data=[list(info) for info in bigram_freq])
        bigrams_df[0] = bigrams_df[0].apply(list).apply(" ".join)
        bigrams_df = bigrams_df.rename(columns={0:"Lemma", 1:"Freq"})
        return bigrams_df

def f_all_comment(comment_grp, vectors, threshold, bigrams_ens): # filters the bigrams of comment using tf idf
    document = np.array([])
    for index, lem in zip(comment_grp['index'], comment_grp['Lemma']): 
        g = BigramCollocationFinder.from_words(lem.split()).score_ngrams(BigramAssocMeasures.raw_freq)
        values = vectors[index].data  # Non-zero values in the sparse matrix
        mask = values >= threshold
        values = values[mask]
        indices = vectors[index].indices[mask]
        keep_bigrams = bigrams_ens[indices[np.argsort(values)[::-1]]]
        kept = np.array([" ".join(bigram) for bigram, _ in g if " ".join(bigram) in keep_bigrams])
        if kept.size != 0:
            document = np.concatenate((document,kept), axis = 0)
    return document   


# 
# ------------------------------------------------------------------------------------ CAS 1 : prediction with type of users
# 

# for CAS 1
def _knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type="simi", k=40):

    user_ind = users_table[users_table == user_id].index[0]
    games_to_hide = np.random.choice(games_to_consider, size=200, replace=False)

    hidden_games = np.intersect1d(games_table[games_table.isin(games_to_hide)].index, mask_ratings[user_ind, :].nonzero()[0])

    prev_ratings, prev_mask_ratings = matrix_ratings[user_ind, :], mask_ratings[user_ind, :], 
    prev_sim = cos_sim_matrix[user_ind, :]

    # hide games
    matrix_ratings[user_ind, hidden_games] = 0
    mask_ratings[user_ind, hidden_games] = 0

    recalc_cos_similarity(user_ind, matrix_ratings, cos_sim_matrix)

    # choice of similar users
    knn_all_user = get_KNN(cos_sim_matrix, users_table.shape[0], user_ind)
    
    match type:
        case 'simi':
            sim_users =  knn_all_user[:k]
        case 'less_simi':
            sim_users = knn_all_user[-k:]
        case 'random':
            sim_users = np.random.choice(knn_all_user, size=k, replace=False)
              
    pred_ratings, mask_pred_ratings = predict_ratings_baseline(matrix_ratings, mask_ratings,
                                                                sim_users, cos_sim_matrix, user_ind)
    
    # restore
    matrix_ratings[user_ind, :], mask_ratings[user_ind, :] = prev_ratings, prev_mask_ratings
    cos_sim_matrix[user_ind, :], cos_sim_matrix[:, user_ind] = prev_sim, prev_sim

    diff = np.abs(matrix_ratings[user_ind, hidden_games] - pred_ratings[hidden_games])

    ALLOW_ERR = 2
    user_mean = users_mean.loc[users_mean["User id"] == user_id, "Rating"].item()
    pos, neg = pred_ratings[hidden_games] < user_mean, pred_ratings[hidden_games] > user_mean

    neg_pred_games = hidden_games[np.argwhere(neg & (diff < ALLOW_ERR)).flatten()]
    pos_pred_games = hidden_games[np.argwhere(pos & (diff < ALLOW_ERR)).flatten()]

    # Find games ids
    neg_pred_games = games_table[games_table.index.isin(neg_pred_games)].values
    pos_pred_games =  games_table[games_table.index.isin(pos_pred_games)].values

    # Find users ids
    sim_users = users_table[users_table.index.isin(sim_users)].values
    sim_users_neg = comments_all[comments_all["Game id"].isin(neg_pred_games) & comments_all["User id"].isin(sim_users)]
    sim_users_pos = comments_all[comments_all["Game id"].isin(pos_pred_games) & comments_all["User id"].isin(sim_users)]
  
    user_pos = comments_all[(comments_all["Game id"].isin(neg_pred_games)) & (comments_all["User id"] == user_id)]
    user_neg = comments_all[(comments_all["Game id"].isin(pos_pred_games)) & (comments_all["User id"] == user_id)]

    return sim_users_neg, sim_users_pos, user_neg, user_pos

""" Plot bigrams intersection """

def _plot_barplots(sim_users_neg, sim_users_pos, user_neg, user_pos, user_id): # user_id
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        # Negatives comments
        bigrams_neg = create_df(BigramCollocationFinder.from_documents(sim_users_neg["Lemma"].str.split().tolist()),
                                BigramAssocMeasures.raw_freq)

        bigrams_neg_user = create_df(BigramCollocationFinder.from_documents(user_neg["Lemma"].str.split().tolist()),
                                BigramAssocMeasures.raw_freq)
        
        bigrams_neg = bigrams_neg[bigrams_neg["Lemma"].isin(bigrams_neg_user["Lemma"])]

        # Find intersection
        bigrams_neg = bigrams_neg[bigrams_neg["Lemma"].isin(bigrams_neg_user["Lemma"])]
        bigrams_neg_user = bigrams_neg_user[bigrams_neg_user["Lemma"].isin(bigrams_neg["Lemma"])]

        # Positive comments
        bigrams_pos = create_df(BigramCollocationFinder.from_documents(sim_users_pos["Lemma"].str.split().tolist()),
                                BigramAssocMeasures.raw_freq)

        bigrams_pos_user = create_df( BigramCollocationFinder.from_documents(user_pos["Lemma"].str.split().tolist()),
                                BigramAssocMeasures.raw_freq)
        
        # Find intersection
        bigrams_pos = bigrams_pos[bigrams_pos["Lemma"].isin(bigrams_pos_user["Lemma"])]
        bigrams_pos_user = bigrams_pos_user[bigrams_pos_user["Lemma"].isin(bigrams_pos["Lemma"])]


        sns.barplot(data=bigrams_neg.sort_values(by="Freq", ascending=False).head(40), y="Lemma", x="Freq", ax=ax1)
        sns.barplot(data=bigrams_pos.sort_values(by="Freq", ascending=False).head(40), y="Lemma", x="Freq", ax=ax2)

        sns.barplot(data=bigrams_neg_user.sort_values(by="Freq", ascending=False).head(40), y="Lemma", x="Freq", ax=ax1, color="r", alpha=0.5)
        sns.barplot(data=bigrams_pos_user.sort_values(by="Freq", ascending=False).head(40), y="Lemma", x="Freq", ax=ax2, color="r", alpha=0.5)

        ax1.set_title(f"Negative bigrams for user {user_id} (id)")
        ax2.set_title(f"Positive bigrams for user {user_id} (id)")
        ax1.tick_params(axis='y', labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        return ax1, ax2

# type : simi, less_simi, random
def knn_comments(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type='simi', k=40):
    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    _plot_barplots(sim_users_neg, sim_users_pos, user_neg, user_pos, user_id)

""" Plot bigrams intersection with TF-IDF filtering"""

def _plot_barplots_tfidf(sim_users_neg, sim_users_pos, user_neg, user_pos, user_id, threshold, vectors, bigrams_ens): # user id 

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        def f_all_comment(comment_grp): # filters the bigrams of comment using tf idf
            document = np.array([])

            for index, lem in zip(comment_grp['index'], comment_grp['Lemma']): 
                g = BigramCollocationFinder.from_words(lem.split()).score_ngrams(BigramAssocMeasures.raw_freq)
                values = vectors[index].data  # Non-zero values in the sparse matrix
                mask = values >= threshold
                values = values[mask]
                indices = vectors[index].indices[mask]
                keep_bigrams = bigrams_ens[indices[np.argsort(values)[::-1]]]
                kept = np.array([" ".join(bigram) for bigram, _ in g if " ".join(bigram) in keep_bigrams])
                if kept.size != 0:
                    document = np.concatenate((document,kept), axis = 0)
            return document        
        
        def filtered_big_df(df): # construct frequency lemma df
            bigrams_comments = df.groupby('User id').apply(f_all_comment,include_groups=False).reset_index(drop=True).values
            if bigrams_comments.size != 0:
                bigrams_comments = np.hstack(bigrams_comments)
            val, count = np.unique(bigrams_comments, return_counts=True)
            count = count/len(bigrams_comments)
            return pd.DataFrame({"Lemma": val, 'Freq': count})
        
        # Negatives comments
        bigrams_neg = filtered_big_df(sim_users_neg)        
        bigrams_neg_user = filtered_big_df(user_neg)

        # Find intersection
        bigrams_neg = bigrams_neg[bigrams_neg["Lemma"].isin(bigrams_neg_user["Lemma"])]
        bigrams_neg_user = bigrams_neg_user[bigrams_neg_user["Lemma"].isin(bigrams_neg["Lemma"])]

        # # Positive comments
        bigrams_pos = filtered_big_df(sim_users_pos)
        bigrams_pos_user = filtered_big_df(user_pos)
        
        # Find intersection
        bigrams_pos = bigrams_pos[bigrams_pos["Lemma"].isin(bigrams_pos_user["Lemma"])]
        bigrams_pos_user = bigrams_pos_user[bigrams_pos_user["Lemma"].isin(bigrams_pos["Lemma"])]

        sns.barplot(data=bigrams_neg.sort_values(by="Freq", ascending=False).head(40), y="Lemma", x="Freq", ax=ax1)
        sns.barplot(data=bigrams_pos.sort_values(by="Freq", ascending=False).head(40), y="Lemma", x="Freq", ax=ax2)

        sns.barplot(data=bigrams_neg_user.sort_values(by="Freq", ascending=False).head(40), y="Lemma", x="Freq", ax=ax1, color="r", alpha=0.5)
        sns.barplot(data=bigrams_pos_user.sort_values(by="Freq", ascending=False).head(40), y="Lemma", x="Freq", ax=ax2, color="r", alpha=0.5)

        ax1.set_title(f"Negative bigrams for user {user_id} (id)")
        ax2.set_title(f"Positive bigrams for user {user_id} (id)")
        ax1.tick_params(axis='y', labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        return ax1, ax2

# type: random, simi, less_simi
def knn_comments_tfidf_plot(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40):    
    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    _plot_barplots_tfidf(sim_users_neg, sim_users_pos, user_neg, user_pos, user_id, threshold, vectors, bigrams_ens)

# ---------------------------------------- TOPX functions

""" Count the number of bigrams in the intersection, no set """

# using tf idf filtering, count number of intersection in neg and pos
def _count_intersect(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx): # user id 

    def f_all_comment(comment_grp): # filters the bigrams of comment using tf idf
        document = np.array([])

        for index, lem in zip(comment_grp['index'], comment_grp['Lemma']): 
            g = BigramCollocationFinder.from_words(lem.split()).score_ngrams(BigramAssocMeasures.raw_freq)
            values = vectors[index].data  # Non-zero values in the sparse matrix
            mask = values >= threshold
            values = values[mask]
            indices = vectors[index].indices[mask]
            keep_bigrams = bigrams_ens[indices[np.argsort(values)[::-1]]]
            kept = np.array([" ".join(bigram) for bigram, _ in g if " ".join(bigram) in keep_bigrams])
            if kept.size != 0:
                document = np.concatenate((document,kept), axis = 0)
        return document        
    
    def filtered_big_df(df): # construct frequency lemma df
        bigrams_comments = df.groupby('User id').apply(f_all_comment,include_groups=False).reset_index(drop=True).values
        if bigrams_comments.size != 0:
            bigrams_comments = np.hstack(bigrams_comments)
        val, count = np.unique(bigrams_comments, return_counts=True)
        # count = count/len(bigrams_comments)
        return pd.DataFrame({"Lemma": val, 'Freq': count}).sort_values(by='Freq', ascending=False) # sorted df

    # Negatives comments
    bigrams_neg = filtered_big_df(sim_users_neg).head(topx) # Take top x bigrams as predicted user's comment
    bigrams_neg_user = filtered_big_df(user_neg)
    # neg_neigh_nb = np.sum(bigrams_neg['Freq'])
    
    # Find intersection
    bigrams_neg = bigrams_neg[bigrams_neg["Lemma"].isin(bigrams_neg_user["Lemma"])]
    bigrams_neg_user = bigrams_neg_user[bigrams_neg_user["Lemma"].isin(bigrams_neg["Lemma"])]

    # avoid error when merge with empty df
    bigrams_neg['Lemma'] = bigrams_neg['Lemma'].astype(str)
    bigrams_neg_user['Lemma'] = bigrams_neg_user['Lemma'].astype(str)

    # clipping bigrams
    intersection_neg = bigrams_neg.merge(bigrams_neg_user, on='Lemma', suffixes=('_neigh', '_user'))
    intersection_neg['Freq_inter'] = intersection_neg[['Freq_neigh', 'Freq_user']].min(axis=1)
    
    # Positive comments
    bigrams_pos = filtered_big_df(sim_users_pos).head(topx) # Take top x bigrams as predicted user's comment
    bigrams_pos_user = filtered_big_df(user_pos)
    # pos_neigh_nb = np.sum(bigrams_pos['Freq'])

    # Find intersection
    bigrams_pos = bigrams_pos[bigrams_pos["Lemma"].isin(bigrams_pos_user["Lemma"])]
    bigrams_pos_user = bigrams_pos_user[bigrams_pos_user["Lemma"].isin(bigrams_pos["Lemma"])]

    bigrams_pos['Lemma'] = bigrams_pos['Lemma'].astype(str)
    bigrams_pos_user['Lemma'] = bigrams_pos_user['Lemma'].astype(str)

    # clipping bigrams                                    
    intersection_pos = bigrams_pos.merge(bigrams_pos_user, on='Lemma', suffixes=('_neigh', '_user'))
    intersection_pos['Freq_inter'] = intersection_pos[['Freq_neigh', 'Freq_user']].min(axis=1)

    # Intersection between correct and predicted bigrams
    inter_neg = np.sum(intersection_neg['Freq_inter'])/len(sim_users_neg) if len(sim_users_neg) else 0
    inter_pos = np.sum(intersection_pos['Freq_inter'])/len(sim_users_pos) if len(sim_users_pos) else 0

    # without norm
    # inter_neg = np.sum(intersection_neg['Freq_inter'])
    # inter_pos = np.sum(intersection_pos['Freq_inter'])
    
    return  inter_pos, inter_neg

def knn_comments_count(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40, topx=None):    
    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    pos_count, neg_count = _count_intersect(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx)
    return pos_count, neg_count
    
""" Recall (ROUGE like), intersection/nb bigrams user, for each game, no set (with clipping)"""

# using tf idf filtering, count number of intersection in neg and pos
def _intersection_ROUGE(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx): # user id 
        # calculate the intersection between user and neighbors bigrams

        def f_all_comment(comment_grp): 
            # filters the bigrams of comment using tf idf
            document = np.array([])

            for index, lem in zip(comment_grp['index'], comment_grp['Lemma']): 
                # applying threshold
                g = BigramCollocationFinder.from_words(lem.split()).score_ngrams(BigramAssocMeasures.raw_freq)
                values = vectors[index].data  # Non-zero values in the sparse matrix
                mask = values >= threshold
                values = values[mask]
                indices = vectors[index].indices[mask]
                keep_bigrams = bigrams_ens[indices[np.argsort(values)[::-1]]]
                kept = np.array([" ".join(bigram) for bigram, _ in g if " ".join(bigram) in keep_bigrams])
                
                if kept.size != 0:
                    document = np.concatenate((document,kept), axis = 0)

            # document contains bigrams comments
            return document        
        
        def filtered_big_df(df): 
            # construct frequency lemma df
            bigrams_comments = df.groupby('User id').apply(f_all_comment,include_groups=False).reset_index(drop=True).values
            if bigrams_comments.size != 0:
                bigrams_comments = np.hstack(bigrams_comments)
            val, count = np.unique(bigrams_comments, return_counts=True)
            # count = count/len(bigrams_comments)
            return pd.DataFrame({"Lemma": val, 'Freq': count}).sort_values(by='Freq',ascending=False)
        

        """RED: intersection = min(ref, pred) / length ref with ref = neighbors, pred = users"""

        # Negatives comments
        bigrams_neg, bigrams_neg_user = filtered_big_df(sim_users_neg).head(topx), filtered_big_df(user_neg)
        neg_user_nb = np.sum(bigrams_neg_user['Freq']) # number of bigrams in neighbors NO SET

        # avoid error when merge with empty df
        bigrams_neg['Lemma'] = bigrams_neg['Lemma'].astype(str)
        bigrams_neg_user['Lemma'] = bigrams_neg_user['Lemma'].astype(str)

        # Find intersection
        bigrams_neg = bigrams_neg[bigrams_neg["Lemma"].isin(bigrams_neg_user["Lemma"])]
        bigrams_neg_user = bigrams_neg_user[bigrams_neg_user["Lemma"].isin(bigrams_neg["Lemma"])]

        # clipping bigrams
        intersection_neg = bigrams_neg.merge(bigrams_neg_user, on='Lemma', suffixes=('_neigh', '_user'))
        intersection_neg['Freq_inter'] = intersection_neg[['Freq_neigh', 'Freq_user']].min(axis=1)

        # Positive comments
        bigrams_pos, bigrams_pos_user = filtered_big_df(sim_users_pos).head(topx), filtered_big_df(user_pos)
        pos_user_nb = np.sum(bigrams_pos_user['Freq'])

        # avoid error when merge with empty df
        bigrams_pos['Lemma'] = bigrams_pos['Lemma'].astype(str)
        bigrams_pos_user['Lemma'] = bigrams_pos_user['Lemma'].astype(str)
            
        # Find intersection
        bigrams_pos = bigrams_pos[bigrams_pos["Lemma"].isin(bigrams_pos_user["Lemma"])]
        bigrams_pos_user = bigrams_pos_user[bigrams_pos_user["Lemma"].isin(bigrams_pos["Lemma"])]

        # clipping bigrams                                    
        intersection_pos = bigrams_pos.merge(bigrams_pos_user, on='Lemma', suffixes=('_neigh', '_user'))
        intersection_pos['Freq_inter'] = intersection_pos[['Freq_neigh', 'Freq_user']].min(axis=1)

        # Proportion of the user's bigrams that are in the intersection WITH CLIP
        neg_prop = np.sum(intersection_neg['Freq_inter'])/neg_user_nb if neg_user_nb else 0
        pos_prop = np.sum(intersection_pos['Freq_inter'])/pos_user_nb if pos_user_nb else 0

        return  pos_prop, neg_prop

# type: random, simi, less_simi
def knn_comments_ROUGE(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40, topx=None):    
    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    pos_prop, neg_prop = _intersection_ROUGE(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx)
    return pos_prop, neg_prop

""" Precision (BLEU like), intersection/nb bigrams neighbors, for each game no set"""

# using tf idf filtering, count number of intersection in neg and pos
def _calc_intersection_BLEU(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx): # user id 
        # calculate the intersection between user and neighbors bigrams
        
        def filtered_big_df(df): 
            # construct frequency lemma df
            bigrams_comments = df.groupby('User id').apply(f_all_comment,include_groups=False).reset_index(drop=True).values
            if bigrams_comments.size != 0:
                bigrams_comments = np.hstack(bigrams_comments)
            val, count = np.unique(bigrams_comments, return_counts=True)
            # count = count/len(bigrams_comments)
            return pd.DataFrame({"Lemma": val, 'Freq': count}).sort_values(by='Freq', ascending=False)
        
        def f_all_comment(comment_grp): 
            # filters the bigrams of comment using tf idf
            document = np.array([])

            for index, lem in zip(comment_grp['index'], comment_grp['Lemma']): 
                # applying threshold
                g = BigramCollocationFinder.from_words(lem.split()).score_ngrams(BigramAssocMeasures.raw_freq)
                values = vectors[index].data  # Non-zero values in the sparse matrix
                mask = values >= threshold
                values = values[mask]
                indices = vectors[index].indices[mask]
                keep_bigrams = bigrams_ens[indices[np.argsort(values)[::-1]]]
                kept = np.array([" ".join(bigram) for bigram, _ in g if " ".join(bigram) in keep_bigrams])
                
                if kept.size != 0:
                    document = np.concatenate((document,kept), axis = 0)

            # document contains bigrams comments
            return document        

        # ---------------------
        # Negatives comments
        bigrams_neg, bigrams_neg_user = filtered_big_df(sim_users_neg).head(topx), filtered_big_df(user_neg)
        neg_neigh_nb = np.sum(bigrams_neg['Freq']) # sum of user, prediction

        bigrams_neg['Lemma'] = bigrams_neg['Lemma'].astype(str)
        bigrams_neg_user['Lemma'] = bigrams_neg_user['Lemma'].astype(str)

        # Find intersection
        bigrams_neg = bigrams_neg[bigrams_neg["Lemma"].isin(bigrams_neg_user["Lemma"])]
        bigrams_neg_user = bigrams_neg_user[bigrams_neg_user["Lemma"].isin(bigrams_neg["Lemma"])]

        intersection_neg = bigrams_neg.merge(bigrams_neg_user, on='Lemma', suffixes=('_neigh', '_user'))
        intersection_neg['Freq_inter'] = intersection_neg[['Freq_neigh', 'Freq_user']].min(axis=1)

        # Positive comments
        bigrams_pos, bigrams_pos_user = filtered_big_df(sim_users_pos).head(topx), filtered_big_df(user_pos)
        pos_neigh_nb = np.sum(bigrams_pos['Freq'])
        
        bigrams_pos['Lemma'] = bigrams_pos['Lemma'].astype(str)
        bigrams_pos_user['Lemma'] = bigrams_pos_user['Lemma'].astype(str)

        # Find intersection
        bigrams_pos = bigrams_pos[bigrams_pos["Lemma"].isin(bigrams_pos_user["Lemma"])]
        bigrams_pos_user = bigrams_pos_user[bigrams_pos_user["Lemma"].isin(bigrams_pos["Lemma"])]

        # clipping bigrams                                    
        intersection_pos = bigrams_pos.merge(bigrams_pos_user, on='Lemma', suffixes=('_neigh', '_user'))
        intersection_pos['Freq_inter'] = intersection_pos[['Freq_neigh', 'Freq_user']].min(axis=1)

        neg_prop = np.sum(intersection_neg['Freq_inter'])/neg_neigh_nb if neg_neigh_nb else 0 
        pos_prop = np.sum(intersection_pos['Freq_inter'])/pos_neigh_nb if pos_neigh_nb else 0
        
        return pos_prop, neg_prop
        # intersection = min(ref, pred) / length pred with ref = user, and pred = neighb

# type: random, simi, less_simi
def knn_comments_BLEU(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40, topx=None):    
    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                        mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    
    score_pos, score_neg = _calc_intersection_BLEU(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx)
    return score_pos, score_neg


# 
# ------------------------------------------------------------------------------------ CAS 2 : prediction with knn, typ users chosen after
# 
 

# for CAS 2
def _knn_sim_neg_pos_cas2(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type="simi", k=40):
    user_ind = users_table[users_table == user_id].index[0]
    games_to_hide = np.random.choice(games_to_consider, size=200, replace=False)

    hidden_games = np.intersect1d(games_table[games_table.isin(games_to_hide)].index, mask_ratings[user_ind, :].nonzero()[0])

    prev_ratings, prev_mask_ratings = matrix_ratings[user_ind, :], mask_ratings[user_ind, :], 
    prev_sim = cos_sim_matrix[user_ind, :]

    # hide games
    matrix_ratings[user_ind, hidden_games] = 0
    mask_ratings[user_ind, hidden_games] = 0

    recalc_cos_similarity(user_ind, matrix_ratings, cos_sim_matrix)

    # prediction with knn   
    knn_all_user = get_KNN(cos_sim_matrix, users_table.shape[0], user_ind)
 
    # CAS 2
    # pred_ratings, mask_pred_ratings = predict_ratings_baseline(matrix_ratings, mask_ratings,
    #                                                             knn_all_user[:k], cos_sim_matrix, user_ind)
    
    # choice of similar users, among those that has rated the game
    match type:
        case 'simi':
            sim_users = knn_all_user[:k]
        case 'less_simi':
            sim_users = knn_all_user[-k:]
        case 'random':
            sim_users = np.random.choice(knn_all_user, size=k, replace=False)

    # CAS 1
    pred_ratings, mask_pred_ratings = predict_ratings_baseline(matrix_ratings, mask_ratings,
                                                                sim_users, cos_sim_matrix, user_ind)
    
    # restore
    matrix_ratings[user_ind, :], mask_ratings[user_ind, :] = prev_ratings, prev_mask_ratings
    cos_sim_matrix[user_ind, :], cos_sim_matrix[:, user_ind] = prev_sim, prev_sim

    diff = np.abs(matrix_ratings[user_ind, hidden_games] - pred_ratings[hidden_games])

    ALLOW_ERR = 2
    user_mean = users_mean.loc[users_mean["User id"] == user_id, "Rating"].item()
    neg, pos = pred_ratings[hidden_games] < user_mean, pred_ratings[hidden_games] > user_mean

    neg_pred_games = hidden_games[np.argwhere(neg & (diff < ALLOW_ERR)).flatten()]
    pos_pred_games = hidden_games[np.argwhere(pos & (diff < ALLOW_ERR)).flatten()]

    # Find games ids
    neg_pred_games = games_table[games_table.index.isin(neg_pred_games)].values
    pos_pred_games =  games_table[games_table.index.isin(pos_pred_games)].values

    # Find users ids
    sim_users = users_table[users_table.index.isin(sim_users)].values
    sim_users_neg = comments_all[comments_all["Game id"].isin(neg_pred_games) & comments_all["User id"].isin(sim_users)]
    sim_users_pos = comments_all[comments_all["Game id"].isin(pos_pred_games) & comments_all["User id"].isin(sim_users)]

    user_neg = comments_all[(comments_all["Game id"].isin(neg_pred_games)) & (comments_all["User id"] == user_id)]
    user_pos = comments_all[(comments_all["Game id"].isin(pos_pred_games)) & (comments_all["User id"] == user_id)]

    return sim_users_neg, sim_users_pos, user_neg, user_pos

# -------------------------------------------- Evaluations, V1: choose comments among neighbors
"""PB HERE START"""
""" Count the number of bigrams in the intersection, avg per game, no set"""
# using tf idf filtering, count number of intersection in neg and pos
def _count_intersect_v(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx): # user id      
        
    def one_game_score(user_com, sim_users_com): # NO SET
        # for one game, list of score for all comment with user
        # user_com one row
    
        user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
        document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens) # neighbors comment filtered

        # clipping
        df_user_big = pd.DataFrame(Counter(user_big).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
        df_document = pd.DataFrame(Counter(document).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)

        if topx:
            df_document = df_document.head(topx) 

        intersection = df_document.merge(df_user_big, on='Bigrams', suffixes=('_neigh', '_user'))
        intersection['Freq_inter'] = intersection[['Freq_neigh', 'Freq_user']].min(axis=1)
        
        return np.sum(intersection['Freq_inter'])/len(sim_users_com) if len(sim_users_com) else 0
              

        # ---------------------

    neg, pos = [], []
    neg_game = user_neg["Game id"].unique()
    pos_game = user_pos["Game id"].unique()

    for game_id in neg_game:
        com_user = user_neg[user_neg['Game id'] == game_id]
        coms_neigh = sim_users_neg[sim_users_neg['Game id'] == game_id]
        
        neg.append(0) if coms_neigh.empty else neg.append(one_game_score(com_user, coms_neigh))
        
    for game_id in pos_game:
        com_user = user_pos[user_pos['Game id'] == game_id]
        coms_neigh = sim_users_pos[sim_users_pos['Game id'] == game_id]

        pos.append(0) if coms_neigh.empty else pos.append(one_game_score(com_user, coms_neigh))

    return np.mean(pos) if pos else 0, np.mean(neg) if neg else 0

# type: random, simi, less_simi
def knn_comments_count_v(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40, topx = None):    
    # CAS 2
    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos_cas2(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
   
    pos_count, neg_count = _count_intersect_v(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx)

    return pos_count, neg_count

# ------------------------------------------------- 

""" Recall (ROUGE like), intersection/nb bigrams user, for each game, no set (with clipping)"""

# using tf idf filtering, count number of intersection in neg and pos
def _intersection_ROUGE_v(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx): # user id 
        def one_game_score(user_com, sim_users_com): # NO SET
            # for one game, list of score for all comment with user
            # user_com one row

            user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
            document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens) # neighbors comment filtered

            # clipping
            df_user_big = pd.DataFrame(Counter(user_big).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = pd.DataFrame(Counter(document).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)

            if topx:
                df_document = df_document.head(topx) 

            intersection = df_document.merge(df_user_big, on='Bigrams', suffixes=('_neigh', '_user'))
            intersection['Freq_inter'] = intersection[['Freq_neigh', 'Freq_user']].min(axis=1)

            return np.sum(intersection['Freq_inter'])/len(user_big) if len(user_big) else 0
                
        # ---------------------

        neg, pos = [], []

        neg_game = user_neg["Game id"].unique()
        pos_game = user_pos["Game id"].unique()

        for game_id in neg_game:
            com_user = user_neg[user_neg['Game id'] == game_id]
            coms_neigh = sim_users_neg[sim_users_neg['Game id'] == game_id]

            neg.append(0) if coms_neigh.empty else neg.append(one_game_score(com_user, coms_neigh))

        for game_id in pos_game:
            com_user = user_pos[user_pos['Game id'] == game_id]
            coms_neigh = sim_users_pos[sim_users_pos['Game id'] == game_id]

            pos.append(0) if coms_neigh.empty else pos.append(one_game_score(com_user, coms_neigh))
        
        return np.mean(pos) if pos else 0, np.mean(neg) if neg else 0

# type: random, simi, less_simi
def knn_comments_ROUGE_v(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40, topx = None):    
   
    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos_cas2(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    pos_prop, neg_prop = _intersection_ROUGE_v(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx)
    return pos_prop, neg_prop

""" Precision (BLEU like), intersection/nb bigrams neighbors, for each game no set"""

# using tf idf filtering, count number of intersection in neg and pos
def _calc_intersection_BLEU_v(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx): # user id 
        # calculate the intersection between user and neighbors bigrams
        
        def one_game_score(user_com, sim_users_com): # NO SET
            # for one game, list of score for all comment with user
            user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
            document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens) # neighbors comment filtered

            # clipping
            df_user_big = pd.DataFrame(Counter(user_big).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = pd.DataFrame(Counter(document).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)

            if topx:
                df_document = df_document.head(topx) 

            intersection = df_document.merge(df_user_big, on='Bigrams', suffixes=('_neigh', '_user'))
            intersection['Freq_inter'] = intersection[['Freq_neigh', 'Freq_user']].min(axis=1)

            return np.sum(intersection['Freq_inter'])/len(document) if len(document) else 0
            
        # ---------------------

        neg, pos = [], []

        neg_game = user_neg["Game id"].unique()
        pos_game = user_pos["Game id"].unique()

        for game_id in neg_game:
            com_user = user_neg[user_neg['Game id'] == game_id]
            coms_neigh = sim_users_neg[sim_users_neg['Game id'] == game_id]

            neg.append(0) if coms_neigh.empty else neg.append(one_game_score(com_user, coms_neigh))

        for game_id in pos_game:
            com_user = user_pos[user_pos['Game id'] == game_id]
            coms_neigh = sim_users_pos[sim_users_pos['Game id'] == game_id]

            pos.append(0) if coms_neigh.empty else pos.append(one_game_score(com_user, coms_neigh))

        return np.mean(pos) if pos else 0, np.mean(neg) if neg else 0

# type: random, simi, less_simi
def knn_comments_BLEU_v(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40, topx = None):    
    
    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos_cas2(user_id, games_to_consider, matrix_ratings,
                                                                        mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    
    score_pos, score_neg = _calc_intersection_BLEU_v(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx)
    return score_pos, score_neg


# ------------------------------------------------- Evaluation V2: choose neighbors among ones that rated games, with percentage, calculate 3 types together (Not following embeddings recommendation)

# similar users among those that have rated the game
def _knn_sim_neg_pos_bis(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, users_mean, k=40):
    user_ind = users_table[users_table == user_id].index[0]
    games_to_hide = np.random.choice(games_to_consider, size=200, replace=False)

    hidden_games = np.intersect1d(games_table[games_table.isin(games_to_hide)].index, mask_ratings[user_ind, :].nonzero()[0])

    prev_ratings, prev_mask_ratings = matrix_ratings[user_ind, :], mask_ratings[user_ind, :], 
    prev_sim = cos_sim_matrix[user_ind, :]

    # hide games
    matrix_ratings[user_ind, hidden_games] = 0
    mask_ratings[user_ind, hidden_games] = 0

    recalc_cos_similarity(user_ind, matrix_ratings, cos_sim_matrix)

    # prediction with knn   
    knn_all_user = get_KNN(cos_sim_matrix, users_table.shape[0], user_ind)
 
    pred_ratings, mask_pred_ratings = predict_ratings_baseline(matrix_ratings, mask_ratings,
                                                                knn_all_user[:k], cos_sim_matrix, user_ind)
    
    # restore
    matrix_ratings[user_ind, :], mask_ratings[user_ind, :] = prev_ratings, prev_mask_ratings
    cos_sim_matrix[user_ind, :], cos_sim_matrix[:, user_ind] = prev_sim, prev_sim

    diff = np.abs(matrix_ratings[user_ind, hidden_games] - pred_ratings[hidden_games])

    ALLOW_ERR = 2
    user_mean = users_mean.loc[users_mean["User id"] == user_id, "Rating"].item()
    neg, pos = pred_ratings[hidden_games] < user_mean, pred_ratings[hidden_games] > user_mean

    neg_pred_games = hidden_games[np.argwhere(neg & (diff < ALLOW_ERR)).flatten()]
    pos_pred_games = hidden_games[np.argwhere(pos & (diff < ALLOW_ERR)).flatten()]

    # Find games ids
    neg_pred_games = games_table[games_table.index.isin(neg_pred_games)].values
    pos_pred_games =  games_table[games_table.index.isin(pos_pred_games)].values

    return pos_pred_games, neg_pred_games, knn_all_user

""" Count the number of bigrams in the intersection, avg per game, no set"""
# using tf idf filtering, count number of intersection in neg and pos
def _count_intersect_v_bis(user_id, neg_pred_games, pos_pred_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, topx, perc=0.3): # user id      
        
    def one_game_score(user_com, sim_users_com): # NO SET
        user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
        document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens) # neighbors comment filtered
        
        if topx:
            df_document = pd.DataFrame(Counter(document).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False).head(topx)
            intersect_document = df_document[df_document['Bigrams'].isin(user_big)]
            return np.sum(intersect_document['Freq'])
            
        # check intersection
        intersect = [big for big in document if big in user_big] # intersection no clipping
        return len(intersect) #/len(sim_users_com) if len(sim_users_com) else 0 #to normalize by number of comments
        # ---------------------

    def score_intersect(game_id):
        # users having rated that game
        id_rated = set(comments_all[comments_all["Game id"] == game_id]['User id'])
        knn_rated = [id for id in knn_all_user if id in id_rated]
        size_neigh = int(len(knn_rated)*perc)

        user_neg = comments_all[(comments_all['Game id'] == game_id) & (comments_all["User id"] == user_id)]
        
        # among rated this game users, choose percentage
        # simi
        sim_users = knn_rated[:size_neigh]
        sim_users_neg = comments_all[(comments_all["Game id"] == game_id) & (comments_all["User id"].isin(sim_users))] 
        score_s = one_game_score(user_neg, sim_users_neg)

        #  less_simi
        sim_users = knn_rated[-size_neigh:]    
        sim_users_neg = comments_all[(comments_all["Game id"] == game_id) & (comments_all["User id"].isin(sim_users))] 
        score_ls = one_game_score(user_neg, sim_users_neg)

        #  random
        sim_users = np.random.choice(knn_rated, size=size_neigh, replace=False)
        sim_users_neg = comments_all[(comments_all["Game id"] == game_id) & (comments_all["User id"].isin(sim_users))] 
        score_r = one_game_score(user_neg, sim_users_neg)

        return score_s, score_ls, score_r

    pos_s, pos_ls, pos_r = [], [], []
    neg_s, neg_ls, neg_r = [], [], []

    for game_id in neg_pred_games:
        s, ls, r  = score_intersect(game_id)
        neg_s.append(s)
        neg_ls.append(ls)
        neg_r.append(r)

    for game_id in pos_pred_games:
        s, ls, r  = score_intersect(game_id)
        pos_s.append(s)
        pos_ls.append(ls)
        pos_r.append(r)

    return [np.mean(l) if l else 0 for l in [neg_s, pos_s, neg_ls, pos_ls, neg_r, pos_r]]

# type: random, simi, less_simi
def knn_comments_count_v_bis(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, threshold = 0, k = 40, topx = None):    
    pos_pred_games, neg_pred_games, knn_all_user = _knn_sim_neg_pos_bis(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, users_mean, k)
    return _count_intersect_v_bis(user_id, neg_pred_games, pos_pred_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, topx)

# ------------------------------------------------- OK

""" Recall (ROUGE like), intersection/nb bigrams user, for each game, no set (with clipping)"""

# using tf idf filtering, count number of intersection in neg and pos
def _intersection_ROUGE_v_bis(user_id, neg_pred_games, pos_pred_games, comments_all, threshold, knn_all_user, vectors, bigrams_ens, topx, perc=0.3): # user id 
        
        def one_game_score(user_com, sim_users_com): # NO SET
            user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
            document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens) # neighbors comment filtered

            # clipping
            df_user_big = pd.DataFrame(Counter(user_big).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = pd.DataFrame(Counter(document).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)

            if topx:
                df_document = df_document.head(topx)

            intersection = df_document.merge(df_user_big, on='Bigrams', suffixes=('_neigh', '_user'))
            intersection['Freq_inter'] = intersection[['Freq_neigh', 'Freq_user']].min(axis=1)

            return np.sum(intersection['Freq_inter'])/len(user_big) if len(user_big) else 0
            
        # ---------------------

        def score_intersect(game_id):
            # users having rated that game
            id_rated = set(comments_all[comments_all["Game id"] == game_id]['User id'])
            knn_rated = [id for id in knn_all_user if id in id_rated]
            size_neigh = int(len(knn_rated)*perc)

            user_neg = comments_all[(comments_all['Game id'] == game_id) & (comments_all["User id"] == user_id)]
            
            # among rated this game users, choose percentage
            # simi
            sim_users = knn_rated[:size_neigh]
            sim_users_neg = comments_all[(comments_all["Game id"] == game_id) & (comments_all["User id"].isin(sim_users))] 
            score_s = one_game_score(user_neg, sim_users_neg)

            #  less_simi
            sim_users = knn_rated[-size_neigh:]    
            sim_users_neg = comments_all[(comments_all["Game id"] == game_id) & (comments_all["User id"].isin(sim_users))] 
            score_ls = one_game_score(user_neg, sim_users_neg)

            #  random
            sim_users = np.random.choice(knn_rated, size=size_neigh, replace=False)
            sim_users_neg = comments_all[(comments_all["Game id"] == game_id) & (comments_all["User id"].isin(sim_users))] 
            score_r = one_game_score(user_neg, sim_users_neg)

            return score_s, score_ls, score_r
        
        pos_s, pos_ls, pos_r = [], [], []
        neg_s, neg_ls, neg_r = [], [], []

        for game_id in neg_pred_games:
            s, ls, r  = score_intersect(game_id)
            neg_s.append(s)
            neg_ls.append(ls)
            neg_r.append(r)

        for game_id in pos_pred_games:
            s, ls, r  = score_intersect(game_id)
            pos_s.append(s)
            pos_ls.append(ls)
            pos_r.append(r)

        return [np.mean(l) if l else 0 for l in [neg_s, pos_s, neg_ls, pos_ls, neg_r, pos_r]]

# type: random, simi, less_simi
def knn_comments_ROUGE_v_bis(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, threshold = 0, k = 40, topx = None):    
    pos_pred_games, neg_pred_games, knn_all_user = _knn_sim_neg_pos_bis(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, users_mean, k)
    return _intersection_ROUGE_v_bis(user_id, neg_pred_games, pos_pred_games, comments_all, threshold, knn_all_user, vectors, bigrams_ens, topx)
    # return pos_prop, neg_prop

""" Precision (BLEU like), intersection/nb bigrams neighbors, for each game no set"""

# using tf idf filtering, count number of intersection in neg and pos
def _calc_intersection_BLEU_v_bis(user_id, neg_pred_games, pos_pred_games, comments_all, threshold, knn_all_user, vectors, bigrams_ens, topx, perc=0.3): # user id 
        # calculate the intersection between user and neighbors bigrams
        
        def one_game_score(user_com, sim_users_com): # NO SET
            # for one game, list of score for all comment with user
            user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
            document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens) # neighbors comment filtered
            
            df_user_big = pd.DataFrame(Counter(user_big).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = pd.DataFrame(Counter(document).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)

            if topx:
                df_document = df_document.head(topx)

            intersection = df_document.merge(df_user_big, on='Bigrams', suffixes=('_neigh', '_user'))
            intersection['Freq_inter'] = intersection[['Freq_neigh', 'Freq_user']].min(axis=1)
            
            return np.sum(intersection['Freq_inter'])/len(document) if len(document) else 0
            
                
        # ---------------------
        def score_intersect(game_id):
            # users having rated that game
            id_rated = set(comments_all[comments_all["Game id"] == game_id]['User id'])
            knn_rated = [id for id in knn_all_user if id in id_rated]
            size_neigh = int(len(knn_rated)*perc)

            user_neg = comments_all[(comments_all['Game id'] == game_id) & (comments_all["User id"] == user_id)]
            
            # among rated this game users, choose percentage
            # simi
            sim_users = knn_rated[:size_neigh]
            sim_users_neg = comments_all[(comments_all["Game id"] == game_id) & (comments_all["User id"].isin(sim_users))] 
            score_s = one_game_score(user_neg, sim_users_neg)

            #  less_simi
            sim_users = knn_rated[-size_neigh:]    
            sim_users_neg = comments_all[(comments_all["Game id"] == game_id) & (comments_all["User id"].isin(sim_users))] 
            score_ls = one_game_score(user_neg, sim_users_neg)

            #  random
            sim_users = np.random.choice(knn_rated, size=size_neigh, replace=False)
            sim_users_neg = comments_all[(comments_all["Game id"] == game_id) & (comments_all["User id"].isin(sim_users))] 
            score_r = one_game_score(user_neg, sim_users_neg)

            return score_s, score_ls, score_r
        
        pos_s, pos_ls, pos_r = [], [], []
        neg_s, neg_ls, neg_r = [], [], []

        for game_id in neg_pred_games:
            s, ls, r  = score_intersect(game_id)
            neg_s.append(s)
            neg_ls.append(ls)
            neg_r.append(r)

        for game_id in pos_pred_games:
            s, ls, r  = score_intersect(game_id)
            pos_s.append(s)
            pos_ls.append(ls)
            pos_r.append(r)

        return [np.mean(l) if l else 0 for l in [neg_s, pos_s, neg_ls, pos_ls, neg_r, pos_r]]

# type: random, simi, less_simi
def knn_comments_BLEU_v_bis(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, threshold = 0, k = 40, topx = None):    
    pos_pred_games, neg_pred_games, knn_all_user = _knn_sim_neg_pos_bis(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, users_mean, k)
    return _calc_intersection_BLEU_v_bis(user_id, neg_pred_games, pos_pred_games, comments_all, threshold, knn_all_user, vectors, bigrams_ens, topx)
    # return score_pos, score_neg


