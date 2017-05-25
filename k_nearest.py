from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

lines = open('data/classification/train.txt').readlines()
(labels, docs) = zip(*[line.split('\t') for line in lines])

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(docs)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(vectors, labels)

test_docs = open('data/classification/test.txt').readlines()
test_vectors = vectorizer.transform(test_docs)
predict_labels = knn.predict(test_vectors)

for doc, label in zip(test_docs, predict_labels):
  print(label + " : " + ' '.join(doc.split(' ')[:20]))

