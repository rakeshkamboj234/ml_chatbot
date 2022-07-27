import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

clean_words = []
intent_tags = []
dataset = []
special_chars = ["?", "!", ",", "."]

def create_pkl_model(intents, special_chars, clean_words, intent_tags):
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            clean_words.extend(word_list)
            dataset.append((word_list, intent["tag"]))
            if intent["tag"] not in intent_tags:
                intent_tags.append(intent["tag"])

    clean_words = [lemmatizer.lemmatize(word) for word in clean_words if word not in special_chars]
    clean_words = sorted(set(clean_words))

    intent_tags = sorted(set(intent_tags))

    pickle.dump(clean_words, open("words.pkl", "wb"))
    pickle.dump(intent_tags, open("classes.pkl", "wb"))
    return

def prep_train_data(dataset):
    train_dataset = []
    output = [0] * len(intent_tags)

    for data in dataset:
        bag = []
        word_patterns = data[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in clean_words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_data = list(output)
        output_data[intent_tags.index(data[1])] = 1
        train_dataset.append([bag, output_data])

    random.shuffle(train_dataset)
    train_dataset = np.array(train_dataset)
    return train_dataset

def chatbot_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    hist = model.fit(
        np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1
    )
    model.save("chatbotmodel.h5", hist)
    return

#create the pkl model
create_pkl_model()
#Preparing training data
train_x = list(prep_train_data(dataset)[:, 0])
train_y = list(prep_train_data(dataset)[:, 1])
# model creation
chatbot_model()