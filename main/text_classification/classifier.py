from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle
import yaml
import tensorflow as tf
from keras import backend as K
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Classifier:
    def __init__(self):
        with open('config.yml', 'rb') as f:
            self.conf = yaml.safe_load(f.read())  
        JSON_PATH = 'text_classification/model/train_vi_11_08/model.json'
        M5_PATH = 'text_classification/model/train_vi_11_08/model.h5'
        TOKENIZER_PATH = 'text_classification/model/train_vi_11_08/tokenizer.pickle'
        # load json and create model
        json_file = open(JSON_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        K.clear_session()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model._make_predict_function()
        self.graph = tf.get_default_graph()
        # load weights into new model
        self.loaded_model._make_predict_function()
        with self.graph.as_default():
            self.loaded_model.load_weights(M5_PATH)
        

        # loading
        with open(TOKENIZER_PATH, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        self.total_possible_outcomes = ['brand_name', 'info', 'index', 'content', 'total', 'thank_you']
    
    def predict(self,text):
        tokens = self.tokenizer.texts_to_sequences([text])
        tokens = pad_sequences(tokens, maxlen=self.conf['TEXT_CLASSIFICATION']['MAX_WORD_COUNT'])
        with self.graph.as_default():
            prediction = self.loaded_model.predict(np.array(tokens))
        i,j = np.where(prediction == prediction.max()) #calculates the index of the maximum element of the array across all axis
        # i->rows, j->columns
        K.clear_session()
        i = int(i)
        j = int(j)
        # print("Text: ",text)
        # print("Result:",self.total_possible_outcomes[j])
        return self.total_possible_outcomes[j],prediction[0][j]


