import nltk
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from unidecode import unidecode

from autocorrect import Speller

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('punkt_tab')

FR_stopwords = stopwords.words("french")

def construction_corpus(avis : pd.DataFrame, taille:int) -> dict:
    """ 
    """

    # Division en 2 dataframe : mal-notés, bien notés
    comments = avis[["Comment title", "Comment body"]].apply(lambda x : " ".join(x.values.astype(str)), axis=1).str.lower().apply(unidecode)
    avis['Comments'] = comments
    good = avis[avis['Rating'] >= 5].drop(["Comment title", "Comment body"],axis=1 )
    bad = avis[avis['Rating'] <= 5].drop(["Comment title", "Comment body"],axis=1 )

    good_tokens = filtering_tokens(good['Comments'], taille, True)
    bad_tokens = filtering_tokens(bad['Comments'], taille, False)

    keep = pd.merge(good_tokens, bad_tokens on='Tokens')
    keep['Freq_G'] = keep['Freq_G'].apply(lambda x : x if x else 0)
    keep['Freq_D'] = keep['Freq_D'].apply(lambda x : x if x else 0)
    keep['Freq'] = keep[['Freq_G', 'FreqD']].apply(lambda x : x[0] + x[1])



    # Dataframe merge on tokens puis somme des fréquences
    # TF-IDF !!! Trie par tfidf décroissant

    # tant que dataframe > taille, enlever


def filtering_tokens(avis : pd.DataFrame, taille, good:bool) -> pd.DataFrame:
    # Avis : Rating, Comments
    avis = avis['Comments']
    # Ponctuation, stopwords

    # Removing punctuation



    # Corrections à apporter :
    # word_tokenizations 
    # Filtrage : on garde pas les digits, retrait des répétitions
    # Filtrage longueur : entre 2 et 27
    # Lemmatization : NOUN, ADJ, VERB
    # Trier par fréquences (enlvelevr les < 5) tant que la taille du dataframe > taille, enlever


































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
    
