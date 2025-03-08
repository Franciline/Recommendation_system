import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

FR_stopwords = stopwords.words("french")

def filtering_txt(text : str) -> list[str]:
    """
    Retourne une version filtrée du texte.
    Parameters :
        text : str
    Returns :
        -> list[str]
    """

    # str to word-list breakdown
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    sentences = [list(filter(lambda x : x not in FR_stopwords, sentence)) for sentence in sentences]

    # Stemming and Lemmatazation
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    stemmed_sentences = [list(map(lambda x : stemmer.stem(x),sentence )) for sentence in sentences]
    lemmatized_sentences = [list(map(lambda x : lemmatizer.lemmatize(x),sentence )) for sentence in stemmed_sentences]

    # Removing punctuation 
    lemmatized_sentences = list(filter(lambda x : x not in string.punctuation, lemmatized_sentences))

    return lemmatized_sentences
    
