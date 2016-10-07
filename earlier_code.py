from bs4 import BeautifulSoup
from nltk.tokenize import WhitespaceTokenizer
from collections import Counter, defaultdict
from math import sqrt, log
import os
import pickle
import re

def normalize(word, keep_capitals):
    """
    Normalize capital letters and punctuation.
    """
    if (keep_capitals and not word == word.upper()) or not keep_capitals:
        word = word.lower()
    word = re.sub(r'[^\w\s]', '', word)
    return word

def make_doc_set(index):
    """
    Create a set containing all documents that
    are present in a given index.
    """
    doc_list = []
    for token in index:
        doc_list.extend(list(index[token]['counter'].elements()))
    return set(doc_list)

def add_frequencies(index):
    """
    Add the document and corpus frequency to a
    word in an inverted index.
    """
    for word in index:
        original_counter = index[word]
        document_frequency = len(index[word])
        corpus_frequency = sum([index[word][document]
                                for document
                                in index[word]])
        index[word] = {"counter" : original_counter,
                       "doc_freq" : document_frequency,
                       "corp_freq" : corpus_frequency
                      }
    return index

def TF_IDF(term, doc_set, index):
    """
    Return the tf-idf weight of a given term.
    """
    # catch off terms that aren't in the index
    badresult = []

    if not term in index:
        for doc in doc_set:
            badresult.append([0, doc])
        return badresult

    rated_docs = []
    IDF = 0
    length_termdocs = len(docs_with_token(index, term))
    if not length_termdocs == 0:
        IDF = log(len(doc_set) / length_termdocs)
    for doc in doc_set:
        if doc in index[term]['counter']:
            rated_docs.append([index[term]['counter'][doc]*IDF,doc])
        else:
            rated_docs.append([0,doc])
    return sorted(rated_docs, reverse=True)

def docs_with_token(index, token):
    """
    Return all documents that contain a
    certain token.
    """
    if token in index:
        return set(index[token]['counter'].elements())
    else:
        return set()

def getKey(list_):
    return list_[1]

def TF_IDF_search(query, doc_set, index):
    """
    For all documents calculate how they
    score compared to the query, by tf-idf weighting.
    """
    dic = {}
    for doc in doc_set:
        dic[doc]=0
    for term in query:
        TFIDF = TF_IDF(term, doc_set, index)
        for elem in TFIDF:
            if elem[0]==0:
                break
            else:
                dic[elem[1]]+=elem[0]
    return sorted(dic.items(), reverse=True, key=getKey)

def euclidean_lengths(doc_set, index):
    """
    Compute for each document its Euclidean length.
    """
    dic={}
    for doc in doc_set:
        dic[str(doc)] = 0
    for key in index:
        for doc in index[key]['counter']:
            dic[str(doc)] += index[key]['counter'][str(doc)]**2
    for key in dic:
        dic[key] = sqrt(dic[key])
    return dic

def cosine_similarity_search(query, doc_set, index):
    """
    For given query, return the top 10 documents using Cosine Similarity
    """
    lengths = euclidean_lengths(doc_set, index)

    # For querylength we assume the query to consist of unique words,
    # for justification check markdown box above
    querylength = sqrt(len(query))

    TF_IDF_results = TF_IDF_search(query, doc_set, index)
    final_result = []
    for result in TF_IDF_results:
        final_result.append([result[0],result[1] / (lengths[result[0]] * querylength)])

    # Sort again
    return sorted(final_result, reverse=True, key=getKey)[:10]

def Okapi_BM25_search(query, doc_set, index):
    """
    For given query, return the top 10 documents using BM25
    """
    k1 = 1.6
    b = 0.75

    lengths = euclidean_lengths(doc_set, index)
    querylength = sqrt(len(query))

    # Calculating the average doc length
    s = 0
    counter = 0
    for key in lengths:
        s += lengths[key]
        counter += 1
    avg_len = s/counter

    result = []
    for doc in doc_set:
        subresult = [str(doc), 0]
        for term in query:
            if term in index:
                if doc in index[term]['counter']:
                    tf = index[term]['counter'][str(doc)]
                    val = (tf * (k1+1)) / (tf + k1 * (1-b+b*(lengths[str(doc)]/avg_len)))
                    subresult[1]+=val
        result.append(subresult)
    return sorted(result, reverse=True, key=getKey)[:10]
