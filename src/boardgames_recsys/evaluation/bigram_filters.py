"""N-gram filtering helpers used by bigram evaluation."""

from collections import Counter

import numpy as np
import pandas as pd
from nltk import bigrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder


def create_df(ngram_finder, ngram_stat):
    bigram_freq = ngram_finder.score_ngrams(ngram_stat)

    bigrams_df = pd.DataFrame(data=[list(info) for info in bigram_freq])
    bigrams_df[0] = bigrams_df[0].apply(list).apply(" ".join)
    bigrams_df = bigrams_df.rename(columns={0: "Lemma", 1: "Freq"})
    return bigrams_df


def f_all_comment(comment_grp, vectors, threshold, bigrams_ens):
    """Filter every comment given a TF-IDF threshold and return kept bigrams."""

    document = np.array([])
    for index, lem in zip(comment_grp["index"], comment_grp["Lemma"], strict=False):
        g = BigramCollocationFinder.from_words(lem.split()).score_ngrams(BigramAssocMeasures.raw_freq)
        values = vectors[index].data
        mask = values >= threshold
        values = values[mask]
        indices = vectors[index].indices[mask]
        keep_bigrams = bigrams_ens[indices[np.argsort(values)[::-1]]]
        kept = np.array([" ".join(bigram) for bigram, _ in g if " ".join(bigram) in keep_bigrams])
        if kept.size != 0:
            document = np.concatenate((document, kept), axis=0)
    return document


def f_all_comment_unig(comment_grp, vectors, threshold, unig_ens, topx=None):
    """Filter every comment given a TF-IDF threshold and return kept unigrams."""

    document = np.array([])
    for index, lem in zip(comment_grp["index"], comment_grp["Lemma"], strict=False):
        g = lem.split()

        values = vectors[index].data
        mask = values >= threshold
        values = values[mask]
        indices = vectors[index].indices[mask]

        keep_unig = unig_ens[indices[np.argsort(values)[::-1]]]
        kept = np.array([unig for unig in g if unig in keep_unig])

        df_kept = pd.DataFrame(Counter(kept).items(), columns=["Unigram", "Freq"]).sort_values(
            by="Freq", ascending=False
        )
        kept = df_kept.head(topx)["Unigram"].unique()

        if kept.size != 0:
            document = np.concatenate((document, kept), axis=0)

    return " ".join(document)


def f_all_comment_llm(comment_grp, vectors, threshold, bigrams_ens):
    """Return all bigrams from every comment for LLM-oriented evaluation."""

    document = np.array([])
    for _index, lem in zip(comment_grp["index"], comment_grp["Lemma"], strict=False):
        coms_bigrams = [" ".join(b) for b in bigrams(lem.split())]
        if len(coms_bigrams) != 0:
            document = np.concatenate((document, coms_bigrams), axis=0)

    return document
