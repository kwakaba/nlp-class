from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

lines = open('data/classification/train.txt').readlines()
(labels, docs) = zip(*[line.split('\t') for line in lines[:1600]])

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(docs)

clf = LogisticRegression()
clf.fit(vectors, labels)

test_docs = open('data/classification/test.txt').readlines()
test_vectors = vectorizer.transform(test_docs)
predict_labels = clf.predict(test_vectors)

for doc, label in zip(test_docs, predict_labels):
  print(label + " : " + ' '.join(doc.split(' ')[:20]))

