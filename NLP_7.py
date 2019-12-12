import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
layers = keras.layers
models = keras.models

data = pd.read_csv("bbc-text.csv")

train_size = int(len(data) * .8)

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

train_cat, test_cat = train_test_split(data['category'], train_size)
train_text, test_text = train_test_split(data['text'], train_size)


encoder = LabelEncoder()
encoder.fit(train_cat)
y_train = encoder.transform(train_cat)
y_test = encoder.transform(test_cat)


num_classes = np.max(y_train) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

mylist=[]
import csv
with open('bbc-text-2.csv') as f:
  reader = csv.reader(f)
  for row in reader:
      a=''.join(row)
      a="".join(l for l in a if l not in string.punctuation)
      a=' '.join([w for w in a.split() if len(w) > 1])
      mylist.append(a)

mylist_2=[]
for i in range(train_size):
    mylist_2.append(clean_str(mylist[i]))


data = {
    'label': y_train.tolist(),
    'text': mylist_2
}
data = pd.DataFrame.from_dict(data)

# Form vocab dictionary
vectorizer = CountVectorizer()
vectorizer.fit_transform(data['text'])
vocab_text = vectorizer.vocabulary_

# Convert text
def convert_text(text):
    text_list = text.split()
    for i in range(text_list.__len__()):
        print(text_list[i])
        print(vocab_text[text_list[i]]+1)

data['text'] = data['text'].apply(convert_text)

# Get X and y matrices
y = np.array(data['label'])
X = np.array(data['text'])



