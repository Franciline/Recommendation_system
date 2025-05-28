
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

from reco_systems.user_game_matrix import *
from reco_systems.CF_knn import *
from reco_systems.evaluation import *
from rouge import Rouge
from sacrebleu import BLEU
from nltk import bigrams


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
    """ Returns dataframe of the results, 
        columns : 'Type' = {"Positive Reviews","Negative Reviews"}, 'User' = {"Similar", "Less Similar", "Random"},
        'count' = result """

    simi = pd.DataFrame({"count" : simi_pos, "User" : "Similar","Type" : "Positive Reviews"})
    random = pd.DataFrame({"count" : random_pos, "User" : "Random","Type" : "Positive Reviews"})
    less_simi = pd.DataFrame({"count" : less_simi_pos, "User" : "Less Similar","Type" : "Positive Reviews"})

    df_posneg = pd.concat([simi, less_simi, random])

    simi = pd.DataFrame({"count" : simi_neg, "User" : "Similar","Type" : "Negative Reviews"})
    random = pd.DataFrame({"count" : random_neg, "User" : "Random","Type" : "Negative Reviews"})
    less_simi = pd.DataFrame({"count" : less_simi_neg, "User" : "Less Similar","Type" : "Negative Reviews"})

    return pd.concat([df_posneg, simi, less_simi, random])

def df_user_type_mean(df_posneg):
    """Returns dataframe with the mean of each category"""
    group_means = df_posneg.groupby(['User', 'Type'])['count'].mean().reset_index()
    group_means['Type'] = group_means['Type'].replace({'Negative Reviews': 'Mean Negative Reviews','Positive Reviews': 'Mean Positive Reviews'})
    user_order = ['Similar', 'Random', 'Less Similar'] 
    group_means['User'] = pd.Categorical(group_means['User'], categories=user_order, ordered=True)
    return group_means

def plot_posnegviolin(data, means, title='', xlabel='', ylabel='', figname='', save = False):
    """Plot the evaluation's results for Type of users, negative and positive, with the mean of each category"""

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
    """Returns mean for each users on nb_iters, for each list in liste"""
    # ! order
    return [np.mean(np.array(sublist).reshape(nb_iters, n_users), axis = 0) for sublist in liste]

def evaluate_big(func, users, nb_iters, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, threshold=0.13, k=40, topx=None):
    """For given users, evaluate the function for 3 types of neighbors, returns dataframed mean results of the users, and the mean of each category."""
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

def f_all_comment(comment_grp, vectors, threshold, bigrams_ens): 
    """Filters every comments given a tf idf threshold. Returns an array containing all the filtered comments"""

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


def f_all_comment_unig(comment_grp, vectors, threshold, unig_ens, topx=None): 
    """Filters every comments given a tf idf threshold. Returns an array containing all the filtered comments"""
    # filter topx too
    document = np.array([])
    for index, lem in zip(comment_grp['index'], comment_grp['Lemma']): 
        g = lem.split()

        values = vectors[index].data  # Non-zero values in the sparse matrix
        mask = values >= threshold
        values = values[mask]
        indices = vectors[index].indices[mask]

        keep_unig = unig_ens[indices[np.argsort(values)[::-1]]]
        kept = np.array([unig for unig in g if unig in keep_unig])

        # topx
        df_kept = pd.DataFrame(Counter(kept).items(), columns=['Unigram', 'Freq']).sort_values(by='Freq', ascending=False)
        kept = df_kept.head(topx)['Unigram'].unique()
        
        if kept.size != 0:
            document = np.concatenate((document,kept), axis = 0)
    
    return " ".join(document)   

def f_all_comment_llm(comment_grp, vectors, threshold, bigrams_ens): 
    """Filters every comments given a tf idf threshold. Returns an array containing all the filtered comments"""

    document = np.array([])
    for index, lem in zip(comment_grp['index'], comment_grp['Lemma']): 
        coms_bigrams = [" ".join(b) for b in bigrams(lem.split())]
        if len(coms_bigrams) != 0:
            document = np.concatenate((document,coms_bigrams), axis = 0)
    
    return document   


# 
# ------------------------------------------------------------------------------------ prediction with type of users
# 

def _knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type="simi", k=40):
    """Returns the comments of a user and its neighbors for well predicted games, user's positive and negative game"""
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
    """Calculate the recall (intersection / nb bigrams in the reference) with reference being the user's comments, and the prediction the neighbors' comments """ 
   
    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    pos_prop, neg_prop = _intersection_ROUGE_v(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx)
    return pos_prop, neg_prop

# ------------------------------------------------- 

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
    """Calculate the precision (intersection / nb bigrams in the prediction) with reference being the user's comments, and the prediction the neighbors' comments """ 

    sim_users_neg, sim_users_pos, user_neg, user_pos = _knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                        mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    
    score_pos, score_neg = _calc_intersection_BLEU_v(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, topx)
    return score_pos, score_neg


# -----------------------------------------------
# Code for final evaluation versions
# 


# on all coms
def _knn_sim(user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, k=40):
    """Returns the ids of well predicted games and the sorted knn on all users"""
    user_ind = users_table[users_table == user_id].index[0]

    # previous 
    prev_ratings, prev_mask_ratings = matrix_ratings[user_ind, :], mask_ratings[user_ind, :], 
    prev_sim = cos_sim_matrix[user_ind, :]

    hidden_games = hide_ratings(matrix_ratings, mask_ratings, user_ind, 0.1) # hide 10 percent of user's game
    recalc_cos_similarity(user_ind, matrix_ratings, cos_sim_matrix)

    # choice of similar users
    knn_all_user = get_KNN(cos_sim_matrix, users_table.shape[0], user_ind)

    # prediction
    pred_ratings, mask_pred_ratings = predict_ratings_baseline(matrix_ratings, mask_ratings,
                                                                knn_all_user[:k], cos_sim_matrix, user_ind)
    hidden_games = np.intersect1d(hidden_games, mask_pred_ratings)

    # restore
    matrix_ratings[user_ind, :], mask_ratings[user_ind, :] = prev_ratings, prev_mask_ratings
    cos_sim_matrix[user_ind, :], cos_sim_matrix[:, user_ind] = prev_sim, prev_sim

    # well predicted games
    ALLOW_ERR = 2
    diff = np.abs(matrix_ratings[user_ind, hidden_games] - pred_ratings[hidden_games])
    
    # for all games
    well_predicted_games = hidden_games[np.argwhere(diff < ALLOW_ERR).flatten()] 
    well_predicted_games = games_table[games_table.index.isin(well_predicted_games)].values

    return well_predicted_games, users_table.loc[knn_all_user].values #true ids of knn

def _intersection_ROUGE(user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, k, topx): 
        
        def one_game_score(user_com, sim_users_com): # NO , bigrams
            # user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
            document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens) # neighbors comment filtered
            
            # clipping
            df_user_big = pd.DataFrame(Counter(user_com).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = pd.DataFrame(Counter(document).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = df_document.head(topx) 
            # print(df_document)

            intersection = df_document.merge(df_user_big, on='Bigrams', suffixes=('_neigh', '_user'))
            intersection['Freq_inter'] = intersection[['Freq_neigh', 'Freq_user']].min(axis=1)
          
            return np.sum(intersection['Freq_inter'])/np.sum(df_user_big['Freq']) *100
        
        def inter(neig, user):
            df_user = pd.DataFrame(Counter(user.split()).items(), columns=['Uni', 'Freq']).sort_values(by='Freq', ascending=False)
            df_neig = pd.DataFrame(Counter(neig.split()).items(), columns=['Uni', 'Freq']).sort_values(by='Freq', ascending=False)
            df_neig = df_neig.head(topx)
            # print("taille", len(df_neig))

            intersection = df_neig.merge(df_user, on='Uni', suffixes=('_neigh', '_user'))
            intersection['Freq_inter'] = intersection[['Freq_neigh', 'Freq_user']].min(axis=1)

            # print("intersection!!!")
            # print(intersection['Uni'].unique())

        # ---------------------
        if len(well_predicted_games) == 0:
            return None, None
        
        score_simi = np.array([0.,0.,0.])
        score_rouge1 = np.array([0.,0.,0.])
        r = Rouge()

        for game_id in well_predicted_games: 
            # print(game_id)
            # users having rated the games
            users_rated = comments_all[comments_all['Game id']==game_id]['User id'].values
            index_uid = np.where(users_rated == user_id)[0][0]  # first occurrence
            users_rated = np.delete(users_rated, index_uid)

            user_real = comments_all[(comments_all['User id'] == user_id) & (comments_all['Game id'] == game_id)]['Comment'].values[0]
            user_com = bigrams(user_real.split())
            user_com = [" ".join(x) for x in user_com]
            
            # - similare 
            m_users = np.intersect1d(knn_all_user[:k], users_rated)
            m = len(m_users)

            neigb_coms = comments_all[(comments_all['User id'].isin(m_users)) & (comments_all['Game id'] == game_id)]
            score_rouge1[0] += r.get_scores(" ".join(neigb_coms['Comment'].values), user_real)[0]['rouge-1']['r']*100
            score_simi[0] += one_game_score(user_com, neigb_coms)
            # print("inter simi")
            # inter(" ".join(neigb_coms['Comment'].values),user_real)

            # - random
            m_random = np.random.choice(users_rated, m, replace=False)
            neigb_coms = comments_all[(comments_all['User id'].isin(m_random)) & (comments_all['Game id'] == game_id)]
            score_rouge1[1] += r.get_scores(" ".join(neigb_coms['Comment'].values), user_real)[0]['rouge-1']['r']*100
            # print("user rand")
            # inter(" ".join(neigb_coms['Comment'].values),user_real)
            score_simi[1] += one_game_score(user_com, neigb_coms)

            # - less similar
            mask = np.isin(knn_all_user, users_rated)
            m_far = knn_all_user[mask][-m:]
            neigb_coms = comments_all[(comments_all['User id'].isin(m_far)) & (comments_all['Game id'] == game_id)]
            score_rouge1[2] += r.get_scores(" ".join(neigb_coms['Comment'].values), user_real)[0]['rouge-1']['r']*100
            # print("user dist")
            # inter(" ".join(neigb_coms['Comment'].values),user_real)
            score_simi[2] += one_game_score(user_com, neigb_coms)

        return score_simi/len(well_predicted_games), score_rouge1/len(well_predicted_games)
                 
# type: random, simi, less_simi
def knn_ROUGE(user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, vectors, bigrams_ens, threshold = 0, k = 40, topx = None):    
    well_predicted_games, knn_all_user = _knn_sim(user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, k)

    score, score_rouge = _intersection_ROUGE(user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, k, topx)
    return score, score_rouge

# ---

# for annexe, return score for each game
def _intersection_ROUGE_annexe(user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, topx, k): 
    
        def one_game_score(user_com, sim_users_com,  v = False, verb = False): # NO , bigrams
            # user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
           
            document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens) # neighbors comment filtered

            # clipping
            df_user_big = pd.DataFrame(Counter(user_com).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = pd.DataFrame(Counter(document).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = df_document.head(topx) 

            intersection = df_document.merge(df_user_big, on='Bigrams', suffixes=('_neigh', '_user'))
            intersection['Freq_inter'] = intersection[['Freq_neigh', 'Freq_user']].min(axis=1)
            
            return np.sum(intersection['Freq_inter'])/np.sum(df_user_big['Freq']) * 100 #len(user_com) if len(user_com) else 0
                
        # ---------------------
        
        if len(well_predicted_games) == 0:
            return [[], [], []], [[], [], []]

        score_simi = [[], [], []] # one user, all score for games with different type of users
        score_rouge1 = [[], [], []]
        r = Rouge()

        for game_id in well_predicted_games: 
            # print(game_id)
            
            # users having rated the games
            users_rated = comments_all[comments_all['Game id']==game_id]['User id'].values
            index_uid = np.where(users_rated == user_id)[0][0]  # first occurrence
            users_rated = np.delete(users_rated, index_uid)

            user_real = comments_all[(comments_all['User id'] == user_id) & (comments_all['Game id'] == game_id)]['Comment'].values[0]
     
            user_com = bigrams(comments_all[(comments_all['User id'] == user_id) & (comments_all['Game id'] == game_id)]['Comment'].values[0].split())
            user_com = [" ".join(x) for x in user_com]

            # - similare 
            m_users = np.intersect1d(knn_all_user[:k], users_rated)
            m = len(m_users)
            neigb_coms = comments_all[(comments_all['User id'].isin(m_users)) & (comments_all['Game id'] == game_id)]
            score_simi[0].append(one_game_score(user_com, neigb_coms))
            score_rouge1[0].append(r.get_scores(" ".join(neigb_coms['Comment'].values), user_real)[0]['rouge-1']['r']*100)

            # - random
            m_random = np.random.choice(users_rated, min(len(users_rated), m), replace=False)
            neigb_coms = comments_all[(comments_all['User id'].isin(m_random)) & (comments_all['Game id'] == game_id)]
            score_simi[1].append(one_game_score(user_com, neigb_coms))
            score_rouge1[1].append(r.get_scores(" ".join(neigb_coms['Comment'].values), user_real)[0]['rouge-1']['r']*100)

            # - less similar
            mask = np.isin(knn_all_user, users_rated)
            m_far = knn_all_user[mask][-m:]
            neigb_coms = comments_all[(comments_all['User id'].isin(m_far)) & (comments_all['Game id'] == game_id)]
            score_simi[2].append(one_game_score(user_com, neigb_coms))
            score_rouge1[2].append(r.get_scores(" ".join(neigb_coms['Comment'].values), user_real)[0]['rouge-1']['r']*100)

        return score_simi, score_rouge1

# type: random, simi, less_simi
def knn_ROUGE_annexe(user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, vectors, bigrams_ens, threshold = 0, k = 40, topx = None):    
    well_predicted_games, knn_all_user = _knn_sim(user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, k)
    liste_score, liste_rouge1 = _intersection_ROUGE_annexe(user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, topx, k)

    return np.array(liste_score), np.array(liste_rouge1)

# with rouge1 tf idf and top x
# on all coms

def _intersection_ROUGE12(user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors_big, bigrams_ens, vector_unig, unig_ens, k, topx): 
        
        def one_game_score(user_com_big, sim_users_com): # NO, bigrams and unigrams
            # user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
            document = f_all_comment(sim_users_com, vectors_big, threshold, bigrams_ens) # neighbors comment filtered
            
            # clipping
            df_user_big = pd.DataFrame(Counter(user_com_big).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = pd.DataFrame(Counter(document).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = df_document.head(topx) 

            intersection = df_document.merge(df_user_big, on='Bigrams', suffixes=('_neigh', '_user'))
            intersection['Freq_inter'] = intersection[['Freq_neigh', 'Freq_user']].min(axis=1)

            predicted = df_document['Bigrams'].unique()
            return np.sum(intersection['Freq_inter'])/np.sum(df_user_big['Freq']), predicted
                
        # ---------------------
        if len(well_predicted_games) == 0:
            return None, None
        
        score_simi = np.array([0.,0.,0.])
        score_rouge1 = np.array([0.,0.,0.])
        r = Rouge()
        maxs, maxr, maxls = -1,-1,-1
        id_gr, id_gs, id_gls = None, None, None


        for game_id in well_predicted_games: 
            # users having rated the games
            users_rated = comments_all[comments_all['Game id']==game_id]['User id'].values
            index_uid = np.where(users_rated == user_id)[0][0]  # first occurrence
            users_rated = np.delete(users_rated, index_uid)

            user_real = comments_all[(comments_all['User id'] == user_id) & (comments_all['Game id'] == game_id)]['Lemma'].values[0]
            user_com_unig = user_real.split()
            user_com = bigrams(user_com_unig)
            user_com = [" ".join(x) for x in user_com]
            
            # - similare 
            m_users = np.intersect1d(knn_all_user[:k], users_rated)
            m = len(m_users)
            neigb_coms = comments_all[(comments_all['User id'].isin(m_users)) & (comments_all['Game id'] == game_id)]
            r2 = one_game_score(user_com, neigb_coms) 
            score_simi[0] += r2[0]* 100
            # rouge 1 on filtered neighbors
            s = r.get_scores(f_all_comment_unig(neigb_coms, vector_unig, threshold, unig_ens), user_real)[0]['rouge-1']['r']*100
            score_rouge1[0] += s
            if r2[0] > maxs : 
                maxs = s
                id_gs = (game_id, r2[1])

            # - random
            m_random = np.random.choice(users_rated, m, replace=False)
            neigb_coms = comments_all[(comments_all['User id'].isin(m_random)) & (comments_all['Game id'] == game_id)]
            s = r.get_scores(f_all_comment_unig(neigb_coms, vector_unig, threshold, unig_ens), user_real)[0]['rouge-1']['r']*100
            score_rouge1[1] += s
            r2 =  one_game_score(user_com, neigb_coms) 
            score_simi[1] += r2[0]* 100
            if r2[0] > maxr : 
                maxr = s
                id_gr = (game_id, r2[1])

            # - less similar
            mask = np.isin(knn_all_user, users_rated)
            m_far = knn_all_user[mask][-m:]
            neigb_coms = comments_all[(comments_all['User id'].isin(m_far)) & (comments_all['Game id'] == game_id)]
            s = r.get_scores(f_all_comment_unig(neigb_coms, vector_unig, threshold, unig_ens), user_real)[0]['rouge-1']['r']*100
            score_rouge1[2] += s
            

            r2 = one_game_score(user_com, neigb_coms) 
            score_simi[2] += r2[0]* 100

            if r2[0] > maxls : 
                maxls = s
                id_gls = (game_id, r2[1])
        
        # print("max :", id_gs, maxs, id_gr, maxr, id_gls, maxls)
        return score_simi/len(well_predicted_games), score_rouge1/len(well_predicted_games)
                 
# type: random, simi, less_simi
def knn_ROUGE12(user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, vectors, bigrams_ens, vector_unig, unig_ens, threshold = 0, k = 40, topx = None):    
    well_predicted_games, knn_all_user = _knn_sim(user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, k)

    score, score_rouge = _intersection_ROUGE12(user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens,vector_unig, unig_ens, k, topx)
    return score, score_rouge


# FOR BLEU PLOT

def _intersection_ROUGE_prim(user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, k, topx): 
        
        def one_game_score(user_com, sim_users_com, v = False): # NO , bigrams
            # user_big = f_all_comment(user_com, vectors, threshold, bigrams_ens)
            document = f_all_comment(sim_users_com, vectors, threshold, bigrams_ens) # neighbors comment filtered
            
            # clipping
            df_user_big = pd.DataFrame(Counter(user_com).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = pd.DataFrame(Counter(document).items(), columns=['Bigrams', 'Freq']).sort_values(by='Freq', ascending=False)
            df_document = df_document.head(topx) 
            # print(df_document)

            intersection = df_document.merge(df_user_big, on='Bigrams', suffixes=('_neigh', '_user'))
            intersection['Freq_inter'] = intersection[['Freq_neigh', 'Freq_user']].min(axis=1)
          
            if v:
                print("doc")
                print(df_document['Bigrams'].unique())
                print(user_com)
            return np.sum(intersection['Freq_inter'])/np.sum(df_user_big['Freq']) 
        
        def mean_bleu(bleu, neigb_coms, user_com):
            # compute mean bleu score for users and neigh
            score = 0
            for hyp in neigb_coms:
                score += bleu.sentence_score(hypothesis=hyp, references=[user_com]).score
            return score/len(neigb_coms) if len(neigb_coms) else 0

        # ---------------------
        if len(well_predicted_games) == 0:
            return None, None, None
        
        score_simi = np.array([0.,0.,0.])
        score_rouge1 = np.array([0.,0.,0.])
        score_bleu = np.array([0.,0.,0.])
        v = False
        r = Rouge()
        bleu = BLEU(max_ngram_order=2,effective_order=True)

        if user_id == 1193 or user_id == 1903:
            v = True

        for game_id in well_predicted_games: 
            if v:
                print(user_id, game_id)
            # users having rated the games
            users_rated = comments_all[comments_all['Game id']==game_id]['User id'].values
            index_uid = np.where(users_rated == user_id)[0][0]  # first occurrence
            users_rated = np.delete(users_rated, index_uid)

            user_real = comments_all[(comments_all['User id'] == user_id) & (comments_all['Game id'] == game_id)]['Lemma'].values[0]
            user_com = bigrams(user_real.split())
            user_com = [" ".join(x) for x in user_com]
            
            # - similare 
            m_users = np.intersect1d(knn_all_user[:k], users_rated)
            m = len(m_users)

            neigb_coms = comments_all[(comments_all['User id'].isin(m_users)) & (comments_all['Game id'] == game_id)]
            hyp = " ".join(neigb_coms['Lemma'].values)
            score_rouge1[0] += r.get_scores(hyp, user_real)[0]['rouge-1']['r']*100
            # score_bleu[0] += mean_bleu(bleu, neigb_coms['Lemma'].values, user_real)
            score_bleu[0] += bleu.sentence_score(hyp, [user_real]).score
            score_simi[0] += one_game_score(user_com, neigb_coms)*100

            # - random
            m_random = np.random.choice(users_rated, m, replace=False)
            neigb_coms = comments_all[(comments_all['User id'].isin(m_random)) & (comments_all['Game id'] == game_id)]
            hyp = " ".join(neigb_coms['Lemma'].values)
            score_rouge1[1] += r.get_scores(hyp, user_real)[0]['rouge-1']['r']*100
            # score_bleu[1] += mean_bleu(bleu, neigb_coms['Lemma'].values, user_real)
            score_bleu[1] += bleu.sentence_score(hyp, [user_real]).score
            score_simi[1] += one_game_score(user_com, neigb_coms)*100

            # - less similar
            mask = np.isin(knn_all_user, users_rated)
            m_far = knn_all_user[mask][-m:]
            neigb_coms = comments_all[(comments_all['User id'].isin(m_far)) & (comments_all['Game id'] == game_id)]
            hyp = " ".join(neigb_coms['Lemma'].values)
            score_rouge1[2] += r.get_scores(hyp, user_real)[0]['rouge-1']['r']*100
            # score_bleu[2] += mean_bleu(bleu, neigb_coms['Lemma'].values, user_real)
            score_bleu[2] += bleu.sentence_score(hyp, [user_real]).score
            score_simi[2] += one_game_score(user_com, neigb_coms)*100

        return score_simi/len(well_predicted_games), score_rouge1/len(well_predicted_games), score_bleu/len(well_predicted_games)
                 
# type: random, simi, less_simi
def knn_ROUGE_prim(user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, vectors, bigrams_ens, threshold = 0, k = 40, topx = None):    
    # no threshold no top x!! bleu score
    well_predicted_games, knn_all_user = _knn_sim(user_id, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, k)

    score, score_rouge, score_bleu = _intersection_ROUGE_prim(user_id, well_predicted_games, comments_all, knn_all_user, threshold, vectors, bigrams_ens, k, topx)
    return score, score_rouge, score_bleu
