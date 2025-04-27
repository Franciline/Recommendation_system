#import nltk
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
#from treetaggerwrapper import TreeTagger
from unidecode import unidecode

#from autocorrect import Speller

#nltk.download("stopwords")
#nltk.download("wordnet")
#nltk.download("omw-1.4")
#nltk.download('punkt_tab')

FR_stopwords = stopwords.words("french")

def words_freq(data) -> pd.DataFrame:
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

def diff_freq(freq_1, freq_2) -> pd.DataFrame:
    """
    Calcule la différence des fréquences
    """

    freq_1 = freq_1[freq_1['Lemma'].isin(freq_2['Lemma'])].copy().sort_values(by='Lemma', ascending=True)
    freq_2 = freq_2[freq_2['Lemma'].isin(freq_1['Lemma'])].copy().sort_values(by='Lemma', ascending=True)

    diff = freq_1['Freq'].to_numpy() - freq_2['Freq'].to_numpy()

    return pd.DataFrame({'Lemma' : freq_1['Lemma'], 'Freq differency' : diff}).sort_values(by='Freq differency',ascending=False)
