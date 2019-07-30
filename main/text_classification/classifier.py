from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle
import yaml

class Classifier:
    def __init__(self,text):
        with open('config.yml', 'rb') as f:
            self.conf = yaml.load(f.read())  

        JSON_PATH = self.conf['TEXT_CLASSIFICATION']['JSON_PATH']
        M5_PATH = self.conf['TEXT_CLASSIFICATION']['M5_PATH']
        TOKENIZER_PATH = self.conf['TEXT_CLASSIFICATION']['TOKENIZER_PATH']
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

        self.text = text
        self.total_possible_outcomes = ['brand_name', 'info', 'index', 'content', 'total', 'thank_you']

    def predict(self):
        tokens = self.tokenizer.texts_to_sequences([self.text])
        tokens = pad_sequences(tokens, maxlen=self.conf['TEXT_CLASSIFICATION']['MAX_WORD_COUNT'])
        prediction = self.loaded_model.predict(np.array(tokens))
        i,j = np.where(prediction == prediction.max()) #calculates the index of the maximum element of the array across all axis
        # i->rows, j->columns
        i = int(i)
        j = int(j)
        print("Text: ",self.text)
        print("Result:",self.total_possible_outcomes[j])
        return self.total_possible_outcomes[j],prediction[0][j]