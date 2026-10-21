from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def run(texts, labels):
    vec = TfidfVectorizer(min_df=2)
    X = vec.fit_transform(texts)  # expect: UL101
    X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.25)
    nb = MultinomialNB().fit(X_tr, y_tr)
    return nb.score(X_te, y_te)
