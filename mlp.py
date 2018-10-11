
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def main():
    mlp = MLPClassifier(alpha=1)
    stop = set(stopwords.words('english'))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


    reader = pd.read_csv('spam.csv', encoding='Latin-1', delimiter=",")

    X_train_test = reader.v2


    train, test = train_test_split(reader, test_size=0.3)

    X_train = train.v2
    Y_train = train.v1

    X_test = test.v2
    Y_test = test.v1

    X_train_list = []
    for row in X_train:
		#print("Progress:", (index+1), "/", self.totalsents)
        row = re.sub("[^a-zA-Z]", " ", row.lower())

		#row = [self.wordnet_lemmatizer.lemmatize(self.porter_stemmer.stem(w)) \
		#for w in self.tokenizer.tokenize(row) if w not in self.stopwords \
        row = [w for w in tokenizer.tokenize(row) if w not in stop if len(w) > 3]
		#row = ' '.join(row)
        print(row)
        X_train_list.append(row)


    tfidf_transformer = TfidfVectorizer()

    clf = LinearSVC().fit(tfidf_transformer.fit_transform(X_train), Y_train)
    predict = clf.predict(tfidf_transformer.transform(X_test))

    print(accuracy_score(Y_test, predict))

    sections = ['ham',"spam"]
    cm = confusion_matrix(Y_test, predict, labels = sections)
    print(cm)




if __name__ == "__main__":
    main()
