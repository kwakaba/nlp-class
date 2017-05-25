from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

lines = open('data/classification/train.txt').readlines()
(labels, docs) = zip(*[line.split('\t') for line in lines])

vectorizer = CountVectorizer(binary=True)
vectors = vectorizer.fit_transform(docs)

dt = DecisionTreeClassifier(criterion="entropy", max_depth=3)
dt.fit(vectors, labels)

export_graphviz(dt, out_file="tree.dot", feature_names=vectorizer.get_feature_names(), class_names=dt.classes_)

test_docs = open('data/classification/test.txt').readlines()
test_vectors = vectorizer.transform(test_docs)
predict_labels = dt.predict(test_vectors)

for doc, label in zip(test_docs, predict_labels):
  print(label + " : " + ' '.join(doc.split(' ')[:20]))

