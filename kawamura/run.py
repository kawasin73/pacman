from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def data_loader(f_name, l_name):
    with open(f_name, mode='r', encoding='utf-8') as f:
        data = list(set(f.readlines()))
        label = [l_name for i in range(len(data))]
        return data, label

XSS_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS_TEST_FILE = 'dataset/test_level_1.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'

STOP_WORDS = ['>']

import re

fmt_tag = "</*[a-zA-Z0-9]+|>"
fmt_symbol = "=|:|;|\"|\\\\\\\\|\\\\|\(|\)|`|&"
fmt_html_escape = "&[a-zA-Z0-9]+;"
fmt = "(%s|%s|%s)" %(fmt_tag, fmt_symbol, fmt_html_escape)

def parse_text(text):
    text = text.lower()
    parsed = re.split(fmt, text.rstrip("\n"))
    # remove white space in head and tail
    parsed = map(lambda x : x.strip(), parsed)
    # remove empty string
    parsed = filter(None, parsed)
    return list(parsed)

def run():
    xss_train_data, xss_train_label = data_loader(XSS_TRAIN_FILE, 'xss')
    xss_test_data, xss_test_label = data_loader(XSS_TEST_FILE, 'xss')
    normal_train_data, normal_train_label = data_loader(NORMAL_TRAIN_FILE, 'normal')
    normal_test_data, normal_test_label = data_loader(NORMAL_TEST_FILE, 'normal')

    X_train = xss_train_data + normal_train_data
    y_train = xss_train_label + normal_train_label
    X_test = xss_test_data + normal_test_data
    y_test = xss_test_label + normal_test_label

    # from sklearn.utils import shuffle
    # X_train, y_train = shuffle(X_train, y_train)

    count_vect = CountVectorizer(tokenizer=parse_text, stop_words=STOP_WORDS)
    X_train_counts = count_vect.fit_transform(X_train)
    X_train_counts.shape

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape

    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    X_new_counts = count_vect.transform(X_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    pred = clf.predict(X_new_tfidf)

    acc_score = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(
        pred, y_test, labels=['xss', 'normal']
    )
    print("acc: \n", acc_score)
    print("confusion matrix: \n", conf_mat)

if __name__ == '__main__':
    run()
