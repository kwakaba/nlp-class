from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

lines = open('data/classification/train.txt').readlines()
(labels, docs) = zip(*[line.split('\t') for line in lines])

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(docs)

knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, vectors, labels, cv=5)
print(scores)
print(scores.mean(), scores.std())

dt = DecisionTreeClassifier(criterion="entropy", max_depth=3)
scores = cross_val_score(dt, vectors, labels, cv=5)
print(scores)
print(scores.mean(), scores.std())

lr = LogisticRegression()
scores = cross_val_score(lr, vectors, labels, cv=5)
print(scores)
print(scores.mean(), scores.std())

