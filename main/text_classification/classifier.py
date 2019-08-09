from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle
import yaml

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Classifier(metaclass=Singleton):
    def __init__(self):
        with open('config.yml', 'rb') as f:
            self.conf = yaml.safe_load(f.read())  
        JSON_PATH = 'text_classification/model/model.json'
        M5_PATH = 'text_classification/model/model.h5'
        TOKENIZER_PATH = 'text_classification/model/tokenizer.pickle'
        # load json and create model
        json_file = open(JSON_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights(M5_PATH)


        # loading
        with open(TOKENIZER_PATH, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        self.total_possible_outcomes = ['brand_name', 'info', 'index', 'content', 'total', 'thank_you']

    def predict(self,text):
        tokens = self.tokenizer.texts_to_sequences([text])
        tokens = pad_sequences(tokens, maxlen=self.conf['TEXT_CLASSIFICATION']['MAX_WORD_COUNT'])
        prediction = self.loaded_model.predict(np.array(tokens))
        i,j = np.where(prediction == prediction.max()) #calculates the index of the maximum element of the array across all axis
        # i->rows, j->columns
        i = int(i)
        j = int(j)
        print("Text: ",text)
        print("Result:",self.total_possible_outcomes[j])
        return self.total_possible_outcomes[j],prediction[0][j]

class InfoClassifier(metaclass=Singleton):
    def __init__(self):
        with open('config.yml', 'rb') as f:
            self.conf = yaml.safe_load(f.read())  
        JSON_PATH = 'text_classification/model/info_model.json'
        M5_PATH = 'text_classification/model/info_model.h5'
        TOKENIZER_PATH = 'text_classification/model/info_tokenizer.pickle'
        # load json and create model
        json_file = open(JSON_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights(M5_PATH)


        # loading
        with open(TOKENIZER_PATH, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        self.total_possible_outcomes = ["title","address","phone","datetime","table","client_no","cashier"]

    def predict(self,text):
        tokens = self.tokenizer.texts_to_sequences([text])
        tokens = pad_sequences(tokens, maxlen=self.conf['TEXT_CLASSIFICATION']['MAX_WORD_COUNT'])
        prediction = self.loaded_model.predict(np.array(tokens))
        i,j = np.where(prediction == prediction.max()) #calculates the index of the maximum element of the array across all axis
        # i->rows, j->columns
        i = int(i)
        j = int(j)
        print("Text: ",text)
        print("Result:",self.total_possible_outcomes[j])
        return self.total_possible_outcomes[j],prediction[0][j]

class TotalClassifier(metaclass=Singleton):
    def __init__(self):
        with open('config.yml', 'rb') as f:
            self.conf = yaml.safe_load(f.read())  
        JSON_PATH = 'text_classification/model/total_model.json'
        M5_PATH = 'text_classification/model/total_model.h5'
        TOKENIZER_PATH = 'text_classification/model/total_tokenizer.pickle'
        # load json and create model
        json_file = open(JSON_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights(M5_PATH)


        # loading
        with open(TOKENIZER_PATH, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        self.total_possible_outcomes = ["rate","repay","tax","total","pay","text_money","discount"]

    def predict(self,text):
        tokens = self.tokenizer.texts_to_sequences([text])
        tokens = pad_sequences(tokens, maxlen=self.conf['TEXT_CLASSIFICATION']['MAX_WORD_COUNT'])
        prediction = self.loaded_model.predict(np.array(tokens))
        i,j = np.where(prediction == prediction.max()) #calculates the index of the maximum element of the array across all axis
        # i->rows, j->columns
        i = int(i)
        j = int(j)
        print("Text: ",text)
        print("Result:",self.total_possible_outcomes[j])
        return self.total_possible_outcomes[j],prediction[0][j]