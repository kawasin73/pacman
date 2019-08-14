import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

def data_loader(f_name, l_name):
    with open(f_name, mode='r', encoding='utf-8') as f:
        data = list(set(f.readlines()))
        label = [l_name for i in range(len(data))]
        return data, label

XSS_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS_TEST_FILE = 'dataset/test_level_2.csv'
XSS2_TRAIN_FILE = 'dataset/train_level_2.csv'
XSS2_TEST_FILE = 'dataset/test_level_1.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'

STOP_WORDS = []

fmt_tag = "</*[a-zA-Z0-9]+|>"
fmt_html_escape = "&[a-zA-Z0-9]+;"
fmt_symbol = "=|:|;|\"|\\\\\\\\|\\\\|\(|\)|`|&"

fmt = "(%s|%s|%s)" %(fmt_tag, fmt_html_escape, fmt_symbol)

def filter_not_script(w):
    return (w[0] != "<") or (w == "<script")

def parse_text(text):
    text = text.lower()
    parsed = re.split(fmt, text.rstrip("\n"))
    # remove white space in head and tail
    parsed = map(lambda x : x.strip(), parsed)
    # remove empty string
    parsed = filter(None, parsed)
    # filter not <script tag
    parsed = filter(filter_not_script, parsed)
    # remove ">"
    parsed = filter(lambda x : x != ">", parsed)
    return list(parsed)

def run():
    xss_train_data, xss_train_label = data_loader(XSS_TRAIN_FILE, 'xss')
    xss_test_data, xss_test_label = data_loader(XSS_TEST_FILE, 'xss')
    xss2_train_data, xss2_train_label = data_loader(XSS_TRAIN_FILE, 'xss')
    xss2_test_data, xss2_test_label = data_loader(XSS_TEST_FILE, 'xss')
    normal_train_data, normal_train_label = data_loader(NORMAL_TRAIN_FILE, 'normal')
    normal_test_data, normal_test_label = data_loader(NORMAL_TEST_FILE, 'normal')

    # X_train = xss_train_data + normal_train_data + xss2_train_data
    # y_train = xss_train_label + normal_train_label + xss2_train_label
    # X_test = xss_test_data + normal_test_data + xss2_test_data
    # y_test = xss_test_label + normal_test_label + xss2_test_label

    X_train = xss_train_data + normal_train_data + xss2_train_data
    y_train = xss_train_label + normal_train_label + xss2_train_label
    X_test = xss_test_data + normal_test_data + xss2_test_data
    y_test = xss_test_label + normal_test_label + xss2_test_label


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
    print("=====================================")
    print(" RESULT")
    print("=====================================")
    print("acc: \n", acc_score)
    print("confusion matrix: \n", conf_mat)
    print()

    # print failed data
    print("=====================================")
    print(" FAILED DATA")
    print("=====================================")
    for x, y, result in zip(X_test, y_test, pred):
        if y != result:
            print('[actual: \"%s\" == predict: \"%s\"] : %r' % (y, result, x))
            print("parsed: ", parse_text(x))
            print()

    # cross validation
    print("=====================================")
    print(" CROSS VALIDATION")
    print("=====================================")
    clf = make_pipeline(CountVectorizer(tokenizer=parse_text, stop_words=STOP_WORDS), TfidfTransformer(), MultinomialNB())
    cross = cross_val_score(clf, X_train + X_test, y_train + y_test, cv=6)
    print(cross)
    print("mean: ", np.mean(cross))

if __name__ == '__main__':
    run()
