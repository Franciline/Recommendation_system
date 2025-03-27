import nltk
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

def construction_corpus(taille:int) -> dict:
    """ 
    Construction d'un corpus à partir d'une BDD de commentaires
    avis.colums = 'Comment title', 'Comment body'

    Retourne df avec mots du corpus et leurs fréquences, les 'taille' plus fréquentes
    """
        
    # Corpus creation from lemmatized dataframe
    lemmas = pd.read_csv("generated_data/lemmas.csv", index_col=0)
    lemmas = lemmas[~lemmas["Lemma"].isna()]
    lemmas = lemmas[lemmas['Part of speech'].isin(['ADJ', 'NOM', "VER:infi","VER:pper", "VER:pres"])]
    lemmas = lemmas['Lemma'].to_numpy()
    
    # Occurencies calculation for each lemma
    lem, occ = np.unique(lemmas, return_counts= True)
    freq_lem = pd.DataFrame({'lemma' : lem, 'freq' : occ})

    
    freq_lem = freq_lem.sort_values(by=['freq'],ascending=False)
    return freq_lem.head(taille)['lemma'].to_numpy()


def construction_corpus_df(df) -> dict:
    """ 
    Construction d'un corpus à partir d'une BDD de commentaires
    avis.colums = 'Comment title', 'Comment body'

    Retourne df avec mots du corpus et leurs fréquences, les 'taille' plus fréquentes
    """
        
    # Corpus creation from lemmatized dataframe
    # keep all verbs
    df = df[df['Part of speech'].isin(['ADJ', 'NOM', 'VERB:infi'])]

    
    # Occurencies calculation for each lemma
    lem, occ = np.unique(df['Lemma'].to_numpy(), return_counts= True)
    freq_lem = pd.DataFrame({'lemma' : lem, 'freq' : occ})

    
    freq_lem = freq_lem.sort_values(by=['freq'],ascending=False)
    return freq_lem
    

def words_freq2(data,corpus) -> pd.DataFrame:
    """
    Construction d'un dataframe avec la fréquence des mots dans un corpus
    """

    # Lemmas recovery
    lemmas = pd.read_csv('generated_data/avis_lemmatized.csv')
    lemmas = lemmas[lemmas.index.isin(data.index)]


    lemmas['Comment'] = lemmas['Comment'].apply(lambda row : row.split())
    tokens = lemmas.explode(column='Comment')

    # Calcul des occurences
    lem, occurences = np.unique(tokens['Comment'].to_numpy(), return_counts=True)

    df = pd.DataFrame({'Lemma' : lem, 'Freq' : occurences})
    df['Freq'] = df['Freq'].apply(lambda val : val/df.size)
    
    # Garder uniquement les lemmas qui appraissent dans le corpus
    return df[df['Lemma'].isin(corpus)]


def diff_freq(freq_1, freq_2) -> pd.DataFrame:
    """
    Calcule la différence des fréquences
    """

    freq_1 = freq_1[freq_1['Lemma'].isin(freq_2['Lemma'])].copy().sort_values(by='Lemma', ascending=True)
    freq_2 = freq_2[freq_2['Lemma'].isin(freq_1['Lemma'])].copy().sort_values(by='Lemma', ascending=True)

    diff = freq_1['Freq'].to_numpy() - freq_2['Freq'].to_numpy()

    return pd.DataFrame({'Lemma' : freq_1['Lemma'], 'Freq differency' : diff}).sort_values(by='Freq differency',ascending=False)



































def filtering_title(title : str, corpus = None) -> dict:
    """Filtre les titres yeeeeeeeee"""
    spell = Speller('fr')
    titles = spell(title)
    titles = nltk.word_tokenize(title,language="french")
    print(titles)
    titles = list(filter(lambda x : x not in FR_stopwords, titles))
    if not corpus : corpus = dict()

    # Stemming and Lemmatazation
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    print("titre avant stem et lemma : ", titles)
    titles = list(map(lambda x : stemmer.stem(x),titles ))
    print(titles)
    titles = list(map(lambda x : lemmatizer.lemmatize(x), titles ))
    print(titles)
    titles = list(filter(lambda x : x not in string.punctuation, titles))

    print(titles)
    checked_words = []
    for word in titles:
        if word not in checked_words:
            if len(corpus) == 0 or word not in list(corpus.keys()) : corpus[word] = 1
            else : corpus[word]+=1
            checked_words.append(word)
    return corpus



def filtering_txt(corpus : dict, text : str) -> dict:
    """
    Retourne une version filtrée du texte.
    Parameters :
        text : str
    Returns :
        -> list[str]
    """

    # str to word-list breakdown
    sentences = nltk.sent_tokenize(text,language="french")
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    for sentence in sentences:
        corpus = filtering_title(sentence, corpus)
    return corpus
    sentences = [list(filter(lambda x : x not in FR_stopwords, sentence)) for sentence in sentences]

    # Stemming and Lemmatazation
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    stemmed_sentences = [list(map(lambda x : stemmer.stem(x),sentence )) for sentence in sentences]
    lemmatized_sentences = [list(map(lambda x : lemmatizer.lemmatize(x),sentence )) for sentence in stemmed_sentences]

    # Removing punctuation 
    lemmatized_sentences = list(filter(lambda x : x not in string.punctuation, lemmatized_sentences))

    checked_words = []
    
    for sentence in lemmatized_sentences:
        for word in sentence:
            if word not in checked_words:
                if word in corpus.keys(): corpus[word]+=1
                else : corpus[word] = 1
                checked_words.append(word)

    return corpus
    
