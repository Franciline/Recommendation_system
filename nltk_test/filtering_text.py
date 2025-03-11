import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from autocorrect import Speller

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('punkt_tab')

FR_stopwords = stopwords.words("french")

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
    
