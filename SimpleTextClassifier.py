
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

news = fetch_20newsgroups(subset='all')

def model(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    return classifier, accuracy

pipeline = Pipeline([('vectorizer', TfidfVectorizer()), 
                     ('classifier', MultinomialNB()),
                     ])
 
classifier, accuracy = model(pipeline, news.data, news.target)
