
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
import nltk
import csv
from sklearn.model_selection import train_test_split


def main():
    mlp = MLPClassifier(alpha=1)

    reader = pd.read_csv('spam.csv', encoding='Latin-1', delimiter=",")

    train, test = train_test_split(reader, test_size=0.3)

    Y_train = [y for y in train.v1]

    print(Y_train)



    '''
    for index,row in enumerate(self.reader):
		#print("Progress:", (index+1), "/", self.totalsents)
		row[6] = re.sub("[^a-zA-Z]", " ", row[6].lower())
		#row[6] = [self.wordnet_lemmatizer.lemmatize(self.porter_stemmer.stem(w)) \
		#for w in self.tokenizer.tokenize(row[6]) if w not in self.stopwords \
		row[6] = [w for w in self.tokenizer.tokenize(row[6]) if w not in self.stopwords and len(w) > 3]
		row[6] = ' '.join(row[6])

		index += 1

		yield row #['data']


    '''


if __name__ == "__main__":
    main()
