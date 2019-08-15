import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def data_loader(f_name, l_name):
    with open(f_name, mode='r', encoding='utf-8') as f:
        data = list(set(f.readlines()))
        label = [l_name for i in range(len(data))]
        return data, label

XSS_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS_TEST_FILE = 'dataset/test_level_1.csv'
XSS2_TRAIN_FILE = 'dataset/train_level_2.csv'
XSS2_TEST_FILE = 'dataset/test_level_2.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'
NORMAL4_TRAIN_FILE = 'dataset/train_level_4.csv'
NORMAL4_TEST_FILE = 'dataset/test_level_4.csv'

STOP_WORDS = []

FMT_URL = "https?://"
FMT_TAG = "</?[a-zA-Z0-9]+|>"
FMT_HTML_ESCAPE = "&[a-zA-Z0-9]+;"
FMT_SYMBOL = "=|:|;|\"|\\\\\\\\|\\\\|\(|\)|`|&|#|,"

FORMAT = "(%s|%s|%s|%s)" %(FMT_URL, FMT_TAG, FMT_HTML_ESCAPE, FMT_SYMBOL)


ZEN = "".join(chr(0xff01 + i) for i in range(94))
HAN = "".join(chr(0x21 + i) for i in range(94))

ZEN2HAN = str.maketrans(ZEN, HAN)

def not_script_tag(w):
    return (w[0] == "<") and (w != "<script") and (w != "</script")

def preprocess_text(text):
    text = text.lower()
    text = text.rstrip("\n")
    text = text.translate(ZEN2HAN)
    # join spaced script tag
    text = re.sub("<\s*s\s*c\s*r\s*i\s*p\s*t", "<script", text)
    text = re.sub("<\s*/\s*s\s*c\s*r\s*i\s*p\s*t", "</script", text)
    # join alert
    text = re.sub("a\s*l\s*e\s*r\s*t", "alert", text)
    return text

def parse_text(text):
    text = preprocess_text(text)
    parsed = re.split(FORMAT, text)
    # remove white space in head and tail
    parsed = map(lambda x : x.strip(), parsed)
    # remove empty string
    parsed = filter(None, parsed)
    # filter not <script tag
    parsed = filter(lambda x : not not_script_tag(x), parsed)
    #  remove ">"
    parsed = filter(lambda x : x != ">", parsed)
    # replace tag
    #   parsed = map(lambda x : ("<tag" if not_script_tag(x) else x), parsed)
    return list(parsed)

def run():
    xss_train_data, xss_train_label = data_loader(XSS_TRAIN_FILE, 'xss')
    xss_test_data, xss_test_label = data_loader(XSS_TEST_FILE, 'xss')
    xss2_train_data, xss2_train_label = data_loader(XSS2_TRAIN_FILE, 'xss')
    xss2_test_data, xss2_test_label = data_loader(XSS2_TEST_FILE, 'xss')
    normal_train_data, normal_train_label = data_loader(NORMAL_TRAIN_FILE, 'normal')
    normal_test_data, normal_test_label = data_loader(NORMAL_TEST_FILE, 'normal')
    normal4_train_data, normal4_train_label = data_loader(NORMAL4_TRAIN_FILE, 'normal')
    normal4_test_data, normal4_test_label = data_loader(NORMAL4_TEST_FILE, 'normal')

    X_train = xss_train_data + normal_train_data + xss2_train_data + normal4_train_data
    y_train = xss_train_label + normal_train_label + xss2_train_label + normal4_train_label
    X_test = xss_test_data + normal_test_data + xss2_test_data + normal4_test_data
    y_test = xss_test_label + normal_test_label + xss2_test_label + normal4_test_label

    tfidf_vectorizer = TfidfVectorizer(tokenizer=parse_text, ngram_range=(3,7))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_train_tfidf.shape

    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    X_new_tfidf = tfidf_vectorizer.transform(X_test)

    pred = clf.predict(X_new_tfidf)

    acc_score = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(
        pred, y_test, labels=['xss', 'normal']
    )
    print("acc: \n", acc_score)
    print("confusion matrix: \n", conf_mat)

    for x, y, result in zip(X_test, y_test, pred):
        if y != result:
            print('actual: %s == predict: %s : %r' % (y, result, x))
            print(parse_text(x))
            print()

    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(TfidfVectorizer(tokenizer=parse_text, ngram_range=(3,7)), MultinomialNB())
    accs = cross_val_score(clf, X_train + X_test, y_train + y_test, cv=5)
    print("avg :", np.mean(accs))
    print("cross: ", accs)

if __name__ == '__main__':
    run()
