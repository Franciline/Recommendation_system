
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.util import ngrams
from unidecode import unidecode
import pandas as pd
from string import punctuation
from nltk import word_tokenize
from treetaggerwrapper import TreeTagger
from typing import Union


def apply_words_limit(text: pd.Series, min_words_nb: int) -> pd.DataFrame:
    punc_to_delete = punctuation
    punc_to_delete = punctuation.replace("\'", "")  # ' can have meaning

    # Translation table for punctuation removal
    trans_table = str.maketrans(punc_to_delete, " " * len(punc_to_delete))

    # Punctuation removed
    nb_words = pd.DataFrame(data={"text": text, "text_clean": text.apply(lambda row: row.translate(trans_table))})
    # nb_words = nb_words.assign(text_clean=nb_words[column].apply(lambda row: row.translate(trans_table)))

    # Calc words nb
    nb_words = nb_words.assign(word_count=nb_words["text_clean"].apply(lambda row: len(row.split())))

    # Apply min limit
    filtered_comments = nb_words[nb_words["word_count"] >= min_words_nb]
    return text[text.index.isin(filtered_comments.index)]


def lemmatize_comments(df: pd.DataFrame, min_words_nb: int, min_word_len: int, max_words_len: int,
                       merge_type: str, treetagger_dir: str):
    """

    merge_type : "comments" or "games_desc"
    """

    if merge_type == "comments":
        df = apply_words_limit(df, min_words_nb, "Comment body")
        text = df[["Comment title", "Comment body"]].apply(
            lambda x: " ".join(x.values.astype(str)), axis=1)

        words, lemmas_df = _lemmatize(text, min_word_len, max_words_len, treetagger_dir)
        words = words.reset_index().rename(columns={"index": "Comment line"}).reset_index(drop=False)
        words_lemmatized = lemmas_df.merge(words, on="Tokens")

        # For words where several lemmas suggestions were made, keep only the first one
        keep_first = words_lemmatized.drop_duplicates(subset=["Tokens"], keep="first")
        words = words.merge(keep_first[["Tokens", "Lemma", "POS"]], on="Tokens", how="left")

        # Delete NaNs
        words = words.dropna(subset=["Lemma"]).drop(columns="index")

        return words

    if merge_type == "games_desc":
        text = df["Description"]
        text.index = df["Game id"]  # reindex

        words, lemmas_df = _lemmatize(text, min_word_len, max_words_len, treetagger_dir)

        words = words.reset_index().rename(columns={"index": "Game id"}).reset_index(drop=False)
        words_lemmatized = lemmas_df.merge(words, on="Tokens")

        # For words where several lemmas suggestions were made, keep only the first one
        keep_first = words_lemmatized.drop_duplicates(subset=["Tokens"], keep="first")
        words = words.merge(keep_first[["Tokens", "Lemma", "POS"]], on="Tokens", how="left")
        # Problem for descriptions but = boire -> replace
        words.loc[(words["Tokens"] == "but") & (words["Lemma"] == "boire"), ["Lemma", "POS"]] = ["but", "NOM"]

        # Delete NaNs
        words = words.dropna(subset=["Lemma"]).drop(columns="index")
        words.loc[:, "Lemma"] = words["Lemma"].apply(unidecode)
        return words


def _lemmatize(text: pd.Series, min_words_len, max_words_len, treetagger_dir) -> pd.DataFrame:
    # Define stopwords
    FR_stopwords = stopwords.words("french")
    FR_stopwords += ['donc', 'alors', 'que', 'qui', 'car', 'parce', 'ca']
    FR_stopwords.remove("ne")
    FR_stopwords.remove("pas")
    print(text)
    text = text.str.lower()
    # N' ... pas -> Ne pas
    text = text.str.replace("n'", "ne ")

    # Remove remaining punctuation
    punc_to_delete = punctuation
    punc_to_delete += "…•"
    trans_table = str.maketrans(punc_to_delete, " " * len(punc_to_delete))
    text = text.str.translate(trans_table)

    # Remove stopwords
    text = text.apply(lambda x: " ".join([word for word in x.split() if word not in FR_stopwords]))

    # Tokenization
    # text_df = text.to_frame().rename(columns={0: "Tokens"})
    words = pd.DataFrame({"Tokens": text.apply(word_tokenize)}).explode("Tokens")

    # Replace characters who which repeats 3 or more times
    words["Tokens"] = words["Tokens"].str.replace(r"(.)\1{2,}", r"\1", regex=True)

    # Remove Hashes (hex like words)
    words = words[~words["Tokens"].str.match(r"^(?=.*\d)[a-z0-9]{20,}$", na=False)]

    # Delete digits and words like 4eme, 10e
    words["Tokens"] = words["Tokens"].str.replace(r"\d+\w*", " ", regex=True)

    # Limit words lengths
    words["Len"] = words["Tokens"].str.len()
    words = words[(words["Len"] > min_words_len) & (words["Len"] < max_words_len)].drop(columns="Len")

    # Lemmatization
    tagger = TreeTagger(TAGDIR=treetagger_dir, TAGLANG="fr")
    all_words = words["Tokens"].unique()

    lemmas_df = pd.DataFrame(data={"Lemma": tagger.tag_text(all_words)})

    # Split return value of TreeTagger into Token (word) | Part of Speech | Lemma
    lemmas_df = lemmas_df["Lemma"].str.split('\t', expand=True).set_axis(["Tokens", "POS", "Lemma"], axis=1)

    # Change 'ne' and 'pas' part of speech
    lemmas_df.loc[lemmas_df["Lemma"].isin(["ne", "pas"]), "POS"] = "NEG"

    return words, lemmas_df


def _join_reviews(words: pd.DataFrame, lemmas_df: pd.DataFrame) -> pd.DataFrame:
    """Function to call to merge lemmas and tokens if lemmatization is done on reviews (comments)"""

    words = words.assign(index=words.index).rename(columns={"index": "Comment line"}).reset_index()
    words = words.reset_index(drop=True).assign(index=words.index)

    words_lemmatized = lemmas_df.merge(words, on="Tokens")

    # For words where several lemmas suggestions were made, keep only the first one
    keep_first = words_lemmatized.drop_duplicates(subset=["Tokens"], keep="first")
    words = words.merge(keep_first[["Tokens", "Lemma", "POS"]], on="Tokens", how="left")

    # Delete NaNs
    words = words.dropna(subset=["Lemma"]).drop(columns="index")

    return words


def simplify_VER_POS(words_lemmatized: pd.DataFrame) -> pd.DataFrame:
    # Verbs endings (infitives)
    regex = r'.*(er|ir|re|oir|dre|ire|aitre|oudre|uire|tir)$'

    # Verbes in infitive forms
    verbes = words_lemmatized["POS"].str.contains("VER:")
    mask = words_lemmatized[(~verbes) | ((verbes) & (words_lemmatized["Lemma"].str.match(regex)))]
    mask.loc[:, "POS"] = mask["POS"].str.replace(r'VER:.*', 'VER', regex=True)
    return mask
