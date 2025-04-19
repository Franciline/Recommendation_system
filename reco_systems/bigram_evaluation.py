
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

from reco_systems.user_game_matrix import *
from reco_systems.CF_knn import *
from reco_systems.evaluation import *

from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

# functions

def create_df(ngram_finder, ngram_stat):
        bigram_freq = ngram_finder.score_ngrams(ngram_stat)

        bigrams_df = pd.DataFrame(data=[list(info) for info in bigram_freq])
        bigrams_df[0] = bigrams_df[0].apply(list).apply(" ".join)
        bigrams_df = bigrams_df.rename(columns={0:"Lemma", 1:"Freq"})
        return bigrams_df


def knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type="simi", k=40):
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
    
    match type:
        case 'simi':
            sim_users =  get_KNN(cos_sim_matrix, k, user_ind)
        case 'less_simi':
            sim_users =  get_KNN(cos_sim_matrix, users_table.shape[0], user_ind)
            sim_users = sim_users[-k:]
        case 'random':
            sim_users =  get_KNN(cos_sim_matrix, users_table.shape[0], user_ind)
            sim_users = np.random.choice(sim_users, size=k, replace=False)
              
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
    # print(pos_pred_games)
    user_pos = comments_all[(comments_all["Game id"].isin(neg_pred_games)) & (comments_all["User id"] == user_id)]
    user_neg = comments_all[(comments_all["Game id"].isin(pos_pred_games)) & (comments_all["User id"] == user_id)]

    return sim_users_neg, sim_users_pos, user_neg, user_pos

""" Plot bigrams intersection """

def plot_barplots(sim_users_neg, sim_users_pos, user_neg, user_pos, user_id): # user_id
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
    sim_users_neg, sim_users_pos, user_neg, user_pos = knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    plot_barplots(sim_users_neg, sim_users_pos, user_neg, user_pos, user_id)

""" Plot bigrams intersection with TF-IDF filtering"""

def plot_barplots_tfidf(sim_users_neg, sim_users_pos, user_neg, user_pos, user_id, threshold, vectors, bigrams_ens): # user id 

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
    sim_users_neg, sim_users_pos, user_neg, user_pos = knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    plot_barplots_tfidf(sim_users_neg, sim_users_pos, user_neg, user_pos, user_id, threshold, vectors, bigrams_ens)

""" Count the number of bigrams in the intersection """

# using tf idf filtering, count number of intersection in neg and pos
def count_intersect(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens): # user id 
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

    return  len(bigrams_pos), len(bigrams_neg)

# type: random, simi, less_simi
def knn_comments_count(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40):    
    sim_users_neg, sim_users_pos, user_neg, user_pos = knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    pos_count, neg_count = count_intersect(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens)
    return pos_count, neg_count

""" Count the number of bigrams in the intersection, divided by the number of neighbors' comments """

# using tf idf filtering, count number of intersection in neg and pos
def count_intersect_norm(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens): # user id 
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

    # normalisation
    norm_intersect_pos = len(bigrams_pos)/len(sim_users_pos) if len(sim_users_pos) else 0 # avoid div by 0
    norm_intersect_neg = len(bigrams_neg)/len(sim_users_neg) if len(sim_users_neg) else 0

    return  norm_intersect_pos, norm_intersect_neg 

# type: random, simi, less_simi
def knn_comments_count_norm(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40):    
    sim_users_neg, sim_users_pos, user_neg, user_pos = knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    pos_count, neg_count = count_intersect_norm(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens)
    return pos_count, neg_count


""" Proportion of user's bigrams in the intersection """

# using tf idf filtering, count number of intersection in neg and pos
def intersection_prop(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens): # user id 
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
            count = count/len(bigrams_comments)
            return pd.DataFrame({"Lemma": val, 'Freq': count})

        def nb_unique_big(df1, df2):
            # returns size of df1 union df2 -> number of distinct bigrams
            return len(pd.unique(pd.concat((df1['Lemma'], df2['Lemma']))))

        # Negatives comments
        bigrams_neg, bigrams_neg_user = filtered_big_df(sim_users_neg), filtered_big_df(user_neg)
        neg_user_nb = len(bigrams_neg_user)

        # Find intersection
        bigrams_neg = bigrams_neg[bigrams_neg["Lemma"].isin(bigrams_neg_user["Lemma"])]
        bigrams_neg_user = bigrams_neg_user[bigrams_neg_user["Lemma"].isin(bigrams_neg["Lemma"])]

        # Positive comments
        bigrams_pos, bigrams_pos_user = filtered_big_df(sim_users_pos), filtered_big_df(user_pos)
        pos_user_nb = len(bigrams_pos_user)

        # print("Pos\nNb unique bigramns ", nb_unique_big(bigrams_pos,bigrams_pos_user))
        # print("Nb unique bigrams in user ", len(bigrams_pos_user))
        # print("Nb unique bigrams in neighbors ", len(bigrams_pos))
            
        # Find intersection
        bigrams_pos = bigrams_pos[bigrams_pos["Lemma"].isin(bigrams_pos_user["Lemma"])]
        bigrams_pos_user = bigrams_pos_user[bigrams_pos_user["Lemma"].isin(bigrams_pos["Lemma"])]

        # count of intersection
        assert (len(bigrams_pos) == len(bigrams_pos_user) and len(bigrams_neg) == len(bigrams_neg_user))

        # Proportion of the user's bigrams that are in the intersection
        neg_prop = len(bigrams_neg)/neg_user_nb if neg_user_nb else 0
        pos_prop = len(bigrams_pos)/pos_user_nb if pos_user_nb else 0
        
        # print("Neg: Percentage fo user's bigrams in neigbors ", neg_prop)
        # print("Pos: Percentage fo user's bigrams in neigbors ", pos_prop)

        return  pos_prop, neg_prop

# type: random, simi, less_simi
def knn_comments_prop(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40):    
    sim_users_neg, sim_users_pos, user_neg, user_pos = knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    pos_prop, neg_prop = intersection_prop(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens)
    return pos_prop, neg_prop

""" Average proportion of user's bigrams in a neighbor's comment """

# using tf idf filtering, count number of intersection in neg and pos
def calc_intersection_norm(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens): # user id 
        # calculate the intersection between user and neighbors bigrams
        
        def filtered_big_df(df): 
            # construct frequency lemma df
            bigrams_comments = df.groupby('User id').apply(f_all_comment,include_groups=False).reset_index(drop=True).values
            if bigrams_comments.size != 0:
                bigrams_comments = np.hstack(bigrams_comments)
            val, count = np.unique(bigrams_comments, return_counts=True)
            count = count/len(bigrams_comments)
            return pd.DataFrame({"Lemma": val, 'Freq': count})
        
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

        def neighbors_comm_norm(user_big, sim_users_com):
            # for each comment, calculate the intersection with user's bigrams
            # normalized by the number of bigrams in the comment
            score = 0

            for index, lem in zip(sim_users_com['index'], sim_users_com['Lemma']): 
                # applying threshold
                g = BigramCollocationFinder.from_words(lem.split()).score_ngrams(BigramAssocMeasures.raw_freq)
                values = vectors[index].data  # Non-zero values in the sparse matrix
                mask = values >= threshold
                values = values[mask]
                indices = vectors[index].indices[mask]
                keep_bigrams = bigrams_ens[indices[np.argsort(values)[::-1]]]
                kept = np.array([" ".join(bigram) for bigram, _ in g if " ".join(bigram) in keep_bigrams]) # bigrams comment filtered 
                comment_size = len(np.unique(kept))

                # intersection user and comment
                intersect = user_big[user_big["Lemma"].isin(kept)]["Lemma"].values
                # print("intersection\n ",intersect)

                score += len(intersect)/comment_size if comment_size else 0

            return score/len(sim_users_com) if len(sim_users_com) else 0               
        
        # negatif
        # calculate the bigram of the user
        bigrams_neg_user = filtered_big_df(user_neg)
        score_neg = neighbors_comm_norm(bigrams_neg_user, sim_users_neg)
    
        # Positive comments
        bigrams_pos_user = filtered_big_df(user_pos)
        score_pos = neighbors_comm_norm(bigrams_pos_user, sim_users_pos)

        return score_pos, score_neg

# type: random, simi, less_simi
def knn_comments_norm(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40):    
    sim_users_neg, sim_users_pos, user_neg, user_pos = knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                        mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    
    score_pos, score_neg = calc_intersection_norm(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens)
    pos_count, neg_count = count_intersect(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens)
    
    # print(f'Number intersection bigrams\nPos {pos_count}\nNeg {neg_count}')
    return score_pos, score_neg, pos_count, neg_count


""" Intersection with predicted user's comment and actual user's comment with no norm by number of comments """

# using tf idf filtering, count number of intersection in neg and pos
def count_intersect_topx_nn(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, x): # user id 

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
        return pd.DataFrame({"Lemma": val, 'Freq': count}).sort_values(by='Freq', ascending=False) # sorted df

    # Negatives comments
    bigrams_neg = filtered_big_df(sim_users_neg)   
    bigrams_neg = bigrams_neg.head(x) # Take top x bigrams as predicted user's comment
    bigrams_neg_user = filtered_big_df(user_neg)
    
    # Find intersection
    bigrams_neg = bigrams_neg[bigrams_neg["Lemma"].isin(bigrams_neg_user["Lemma"])]
    
    # Positive comments
    bigrams_pos = filtered_big_df(sim_users_pos)
    bigrams_pos = bigrams_pos.head(x) # Take top x bigrams as predicted user's comment
    bigrams_pos_user = filtered_big_df(user_pos)

    # Find intersection
    bigrams_pos = bigrams_pos[bigrams_pos["Lemma"].isin(bigrams_pos_user["Lemma"])]

    # Intersection between correct and predicted bigrams
    inter_pos = len(bigrams_pos)
    inter_neg = len(bigrams_neg)
    
    return  inter_pos, inter_neg

def knn_comments_count_topx_nn(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40, x=20):    
    sim_users_neg, sim_users_pos, user_neg, user_pos = knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    pos_count, neg_count = count_intersect_topx_nn(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, x)
    return pos_count, neg_count

""" Intersection with predicted user's comment and actual user's comment, with norm by number of comments """

# using tf idf filtering, count number of intersection in neg and pos
def count_intersect_topx(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, x): # user id 

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
        return pd.DataFrame({"Lemma": val, 'Freq': count}).sort_values(by='Freq', ascending=False) # sorted df

    # Negatives comments
    bigrams_neg = filtered_big_df(sim_users_neg)   
    bigrams_neg = bigrams_neg.head(x) # Take top x bigrams as predicted user's comment
    bigrams_neg_user = filtered_big_df(user_neg)
    
    # Find intersection
    bigrams_neg = bigrams_neg[bigrams_neg["Lemma"].isin(bigrams_neg_user["Lemma"])]
    
    # Positive comments
    bigrams_pos = filtered_big_df(sim_users_pos)
    bigrams_pos = bigrams_pos.head(x) # Take top x bigrams as predicted user's comment
    bigrams_pos_user = filtered_big_df(user_pos)

    # Find intersection
    bigrams_pos = bigrams_pos[bigrams_pos["Lemma"].isin(bigrams_pos_user["Lemma"])]

    # Intersection between correct and predicted bigrams
    inter_pos = len(bigrams_pos)/len(bigrams_pos_user) if len(bigrams_pos_user) else 0
    inter_neg = len(bigrams_neg)/len(bigrams_neg_user) if len(bigrams_neg_user) else 0
    
    return  inter_pos, inter_neg

def knn_comments_count_topx(user_id, games_to_consider, matrix_ratings, mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, vectors, bigrams_ens, type = 'simi', threshold = 0, k = 40, x=20):    
    sim_users_neg, sim_users_pos, user_neg, user_pos = knn_sim_neg_pos(user_id, games_to_consider, matrix_ratings,
                                                                                mask_ratings, cos_sim_matrix, users_table, games_table, comments_all, users_mean, type, k)
    pos_count, neg_count = count_intersect_topx(sim_users_neg, sim_users_pos, user_neg, user_pos, threshold, vectors, bigrams_ens, x)
    return pos_count, neg_count
