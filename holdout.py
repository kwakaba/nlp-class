from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

lines = open('data/classification/train.txt').readlines()
(labels, docs) = zip(*[line.split('\t') for line in lines])

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(docs)

v_train, v_test, l_train, l_test = train_test_split(vectors, labels, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(v_train, l_train)

print(classification_report(l_test, knn.predict(v_test)))

dt = DecisionTreeClassifier(criterion="entropy", max_depth=3)
dt.fit(v_train, l_train)

print(classification_report(l_test, dt.predict(v_test)))

lr = LogisticRegression()
lr.fit(v_train, l_train)

print(classification_report(l_test, lr.predict(v_test)))

