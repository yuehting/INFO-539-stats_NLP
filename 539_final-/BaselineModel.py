from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# import dataset
df = pd.read_csv('IMDB Dataset.csv')
print(df)

train = df.iloc[:25000]
test = df.iloc[25000:]

x = train['review']
y = train['sentiment']


tfidf = TfidfVectorizer(ngram_range=(1,2))
x = tfidf.fit_transform(x)
print(x.shape)

train_texts, dev_texts, train_labels,dev_labels = train_test_split(x,y , test_size=.2, random_state = 11, stratify=y)
print(train_texts.shape)
print(dev_texts.shape)

model = LogisticRegression()
model.fit(train_texts,train_labels)
predictions = model.predict(dev_texts)
print(classification_report(dev_labels, predictions))

from sklearn.metrics import accuracy_score
print('predictive accuracy:' ,accuracy_score(dev_labels,predictions))


