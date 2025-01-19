import nltk
from nltk.stem.porter import *
import re
import string
import numpy as np
import pandas as pd
#from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import load_dataset

ds = load_dataset("ruanchaves/hatebr")

#nltk.download('rslp')

stopwords_pt =stopwords = nltk.corpus.stopwords.words('portuguese')
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer_pt = nltk.stem.RSLPStemmer()

def preprocess(text_string):
    space_pattern = r'\s+'
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    mention_regex = r'@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(url_regex, 'URLAQUI', parsed_text)
    parsed_text = re.sub(mention_regex, '', "MENCAO", parsed_text)
    return parsed_text

def tokenize(text):
    text = " ".join(re.split("[^a-zA-Z]*", text.lower())).strip()
    tokens = [stemmer_pt.stem(t) for t in text.split() if t not in stopwords_pt]
    return tokens

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    max_features=1000,
    ngram_range=(1, 3),
    stop_words=stopwords_pt,
    decode_error='replace',
)

tfidf = vectorizer.fit_transform(ds['train']['instagram_comments'])
vocab = { v:i for i, v in enumerate(vectorizer.get_feature_names_out()) }

def vectorize(tokens):
    total = len(tokens)
    vec = np.zeros(len(vocab))
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1 / total
    return vec

pos_vect = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    max_features=1000,
    ngram_range=(1, 3),
    stop_words=stopwords_pt,
    decode_error='replace',
)
