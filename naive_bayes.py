from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from scipy.misc import logsumexp

lines = open('data/classification/train.txt').readlines()
(labels, docs) = zip(*[line.split('\t') for line in lines])

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(docs)

nb = MultinomialNB()
nb.fit(vectors, labels)

test_docs = open('data/classification/test.txt').readlines()
test_vectors = vectorizer.transform(test_docs)
predict_labels = nb.predict(test_vectors)

for doc, label in zip(test_docs, predict_labels):
  print(label + " : " + ' '.join(doc.split(' ')[:20]))


lifts = nb.feature_log_prob_ - logsumexp(nb.feature_log_prob_, axis=0)
fnames = vectorizer.get_feature_names()
for c in range(len(nb.classes_)):
    informative_features = sorted(zip(lifts[c], fnames), reverse=True)
    print("Most informative features of " + nb.classes_[c] + ":")
    print(informative_features[:30])

