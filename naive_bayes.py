from __future__ import division
from collections import Counter, defaultdict
from math import log
from nltk.tokenize import WhitespaceTokenizer
import pandas as pd
import codecs
import re

def get_corpus(frame, n):
    """
    Extract the normalised documents from the top n most common normalised ministries
    and return a list of the documents, ministries, and the top n classes.
    """
    # Extract all documents and their ministries from the pandas Dataframe
    ministeries = list(frame.ministerie)
    documents = list(frame.vraag)

    # Now we extract our top classes and the synonyms per class
    synonyms, classes = normalize_names(ministeries, n)

    # Find all documents that belong to the top classes and normalize all the words in them
    # and at the same time store the corresponding normalised ministry in a seperate list.
    new_d, new_m = zip(*((normalize(x.encode('utf-8')).split(),synonyms[y])
                             for (x,y) in zip(documents, ministeries)
                                 if isinstance(y,(str,unicode)) and synonyms[y] in classes))

    return (new_d, new_m, classes)

def normalize(document):
    """
    Normalize capital letters and punctuation in  a document.
    """
    document = re.sub(r'[^\w\s]', '', document.lower())
    return document

def normalize_names(items, n):
    """
    Normalizes names of given items and returns a counter
    with the names and the number of occurences,
    and a dict containing all equivalences.
    """
    count_items = Counter()
    synonyms = dict()

    for name in items:
        if isinstance(name, (str, unicode)):
            new_name = re.sub('\(.*?\)', '', name.lower()).strip()
            count_items[new_name] += 1
            synonyms[name] = new_name

    count_items = count_items.most_common(n)
    classes = list(zip(*count_items)[0])

    return (synonyms, classes)

def train_multinomial_NB(classes, documents, ministeries):
    """
    Train a Naive Bayes classifier on the given data for
    a given set of classes. Return the vocabulary, and
    prior and conditional probabilities.
    """

    # Extract the vocabulary from the documents.
    vocabulary = set(token for document in documents for token in document)

    n = len(documents)
    class_members = calculate_membership(classes, ministeries)

    cond_probs = {term : dict() for term in vocabulary}
    prior = dict()

    for c in classes:
        nc = len(class_members[c])
        # Calculate the prior for each class
        prior[c] = nc / n
        # Get all the words from a specific class
        text = [word for i in class_members[c] for word in documents[i]]

        # Get the term frequency per term for documents in the class.
        occurences = Counter()
        for token in text:
            occurences[token] += 1

        # Calculate the cond_probability per term per class.
        denominator = sum(occurences[term] for term in occurences) + 1
        for term in vocabulary:
            cond_probs[term][c] = (occurences[term] + 1) / denominator

    return (vocabulary, prior, cond_probs)


def calculate_membership(classes, ministeries):
    """
    Create a dictionary with all indices of class members,
    per class.
    """

    membership = {c : [] for c in classes}
    for i, ministerie in enumerate(ministeries):
        membership[ministerie].append(i)
    return membership

def apply_multinomial_NB(classes, vocabulary, prior, cond_prob, doc):
    """
    Classify doc with the the trained multinomial Naive Bayes classifier,
    which consists of the classes, documents, prior, and cond_prob.
    The funciton returns a predicted class for the document.
    """
    score = {}
    tokens = set([word for word in doc if word in vocabulary])

    for c in classes:
        score[c] = log(prior[c],10)
        for t in tokens:
            score[c] += log(cond_prob[t][c],10)
    # return the class for which the score is maximum.
    return max(score, key=score.get)

def get_occurence_counts(docs, ministeries):
    """
    Returns a dict that for each term has the amount
    of documents it appears in per class/ministerie.
    """
    count_dict = defaultdict(Counter)
    for doc,mini in zip(docs, ministeries):

        # remove duplicates
        doc = set(doc)
        for token in doc:
            count_dict[token][mini] += 1
            count_dict[token]['total'] += 1

    # returning a defaultdict is asking for trouble. Thus, return normal dict
    return dict(count_dict)

def mutual_inf(term, ministerie, count_dict, N, docs_in_class):
    """
    Returns the mutual information of term and ministerie.
    """
    if term not in count_dict:
        return 0
    N11 = count_dict[term].get(ministerie, 0)
    N1_ = count_dict[term].get('total', 0)
    N10 = N1_ - N11
    N01 = docs_in_class - N11
    N0_ = N - N1_
    N00 = N0_ - N01
    N_1 = docs_in_class
    N_0 = N - N_1

    result = 0
    result+=(N11/N) * my_log_and_div(N*N11,N1_*N_1)
    result+=(N01/N) * my_log_and_div(N*N01,N0_*N_1)
    result+=(N10/N) * my_log_and_div(N*N10,N1_*N_0)
    result+=(N00/N) * my_log_and_div(N*N00,N0_*N_0)
    return result

def my_log_and_div(n, m):
    """
    Returns log of n/m with base 2,
    except it returns 0 for n=0.
    """
    if n == 0:
        return 0
    else:
        return log(n/m,2)

def get_mutual_infs(docs, ministeries):
    """
    Returns a nested dict matching terms and classes
    with their mutual information.
    """
    count_dict = get_occurence_counts(docs, ministeries)
    N = len(docs)

    classes = set(ministeries)
    result = {}
    for mini in classes:
        docs_in_class = ministeries.count(mini)
        result[mini] = {}
        for term in count_dict:
            result[mini][term] = mutual_inf(term, mini, count_dict, N, docs_in_class)
    return result

def top_mutual_infs(mutual_infs,n):
    """
    Given the mutual information and interger n,
    return a dict of top n words based on the mutual information.
    """
    ministeries = {}
    for mini in mutual_infs:
        lst = [(mutual_infs[mini][term],term) for term in mutual_infs[mini]]
        ministeries[mini] = sorted(lst, reverse=True)[:n]

    return ministeries

def test_NB(classes, vocab, prior, cond_prob, documents, ministeries):
    """
    Test the Naive Bayes classifier on given test data
    and return a list of classifications for the test documents
    """
    pred_classes = []

    for i, doc in enumerate(documents):
        score = apply_multinomial_NB(classes, vocab, prior, cond_prob, doc)
        pred_classes.append(score)

    return pred_classes

def distribution(classes, test_mins, pred_classes):
    """
    Calculate the True Positives, False Positives, and False Negatives per class
    and return a dict with a counter as value for each class.
    """
    data = { c : Counter() for c in classes }

    for c in classes:
        for i, ministerie in enumerate(test_mins):
            if (c == ministerie and c == pred_classes[i]):
                data[c]['TP'] += 1
            elif (c == pred_classes[i] and c != ministerie):
                data[c]['FP'] += 1
            elif (c != pred_classes[i] and c == ministerie):
                data[c]['FN'] += 1

    return data

def recall(data):
    """
    Calculates the recall per class from the data
    structure returned by the distribution function
    """
    recall = { c : float(data[c]['TP'])/float(data[c]['TP'] + data[c]['FN'])
               for c in data
               if data[c]['TP'] + data[c]['FN'] != 0 }

    for c in data:
        if not c in recall:
            recall[c] = 0

    return recall

def precision(data):
    """
    Calculates the precision per class from the data
    structure returned by the distribution function.
    """
    precision = { c : float(data[c]['TP'])/float(data[c]['TP'] + data[c]['FP'])
                  for c in data
                  if data[c]['TP'] + data[c]['FP'] != 0 }

    for c in data:
        if not c in precision:
            precision[c] = 0
    return precision

def f1(data, classes):
    """
    Calculate the micro-averaged F1 value from the data
    structure returned by the distribution function.
    """
    TP = sum([data[c]['TP'] for c in classes])
    FP = sum([data[c]['FP'] for c in classes])
    FN = sum([data[c]['FN'] for c in classes])

    p = float(TP)/float(TP+FP)
    r = float(TP)/float(TP+FN)
    return 2.0*((p*r)/(p+r))

def split_data(documents, ministeries):
    n = int(0.75 * len(documents))
    train_docs = documents[:n]
    train_mins = ministeries[:n]
    test_docs = documents[n:]
    test_mins = ministeries[n:]
    return (train_docs, train_mins, test_docs, test_mins)

def open_csv(website):
    return pd.read_csv(website,
                   compression='gzip', sep='\t', encoding='utf-8',
                   index_col=0, names=['jaar', 'partij','titel','vraag',
                   'antwoord','ministerie'])
