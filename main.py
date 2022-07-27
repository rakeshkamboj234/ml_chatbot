import random
import json
import pickle
import numpy as np

from flask import Flask, render_template, request
 
app = Flask(__name__)

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
from tensorflow.keras.models import load_model


lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

clean_words = pickle.load(open('words.pkl', 'rb'))
intent_tags = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_sentence(sentence):
    s_words = nltk.word_tokenize(sentence)
    s_words = [lemmatizer.lemmatize(word)  for word in s_words]
    return s_words

def word_collection(sentence):
    s_words= clean_sentence(sentence)
    collection = [0] * len(clean_words)
    for w in s_words:
        for i, word in enumerate(clean_words):
            if word == w:
                collection[i] = 1

    return np.array(collection)

def predict_class(sentence):
    no_of_words = word_collection(sentence)
    res = model.predict(np.array([no_of_words]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda  x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': intent_tags[r[0]], 'probability': str(r[1])})
    return return_list


def fetch_response(intents_list,json_data):
    tag = intents_list[0]['intent']
    intents_list =json_data['intents']
    for i in intents_list:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


@app.route("/")
def home():
    return render_template("index.html")
 
@app.route("/get")
def get_bot_response():
    message = request.args.get('msg')
    message = message.lower() if message else None
    
    if message is None:
        res = "please provide input."
    else:
        ints = predict_class(message)
        res = fetch_response(ints, intents)
        
    return str(res)
 
 
if __name__ == "__main__":
    app.run(debug=True)
    
