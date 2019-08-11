#!/usr/bin/env python
# coding: utf-8

# In[471]:


from __future__ import print_function, division

from keras.datasets import mnist
(X_train, y_train), (_, _) = mnist.load_data()


# In[472]:



print(X_train[0].shape)


# # Get receipt dataset

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence

df_col = ["sentence","brand_name","info","index","content","total","thank_you"]
y_col = ["brand_name","info","index","content","total","thank_you"]
train_df = pd.read_csv('../../text_classification/31-07-vigroupped.csv',   encoding='utf-8')

seed = 120
np.random.seed(seed)
train_df = shuffle(train_df)
train_df.head()

X_train = train_df["sentence"].fillna("fillna").values
Y_train = train_df[['brand_name', 'info', 'index', 'content', 'total', 'thank_you']].values

tokenizer = Tokenizer()
texts = X_train

tokenizer.fit_on_texts(texts) 
Tokenizer_vocab_size = len(tokenizer.word_index) + 1

X_train_encoded_words = tokenizer.texts_to_sequences(X_train)

maxWordCount= 16
maxDictionary_size=Tokenizer_vocab_size
X_train_encoded_padded_words = sequence.pad_sequences(X_train_encoded_words, maxlen=maxWordCount)


# In[4]:


X_train_encoded_padded_words.shape
# Y_train


# In[475]:


tokenizer.word_index['ngừ']


# In[9]:


# Convert Y_train
# range(0,len(y_col))[5]
y_train = []
for row in Y_train:
    for index,col in enumerate(range(0,len(y_col))):
        if row[col] == 1:
            y_train.append(index)
print(len(y_train))


# In[10]:


reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

def padded_sequence_to_text(int_arr):
    
    padded_sequence = int_arr.reshape((maxWordCount))
    padded_sequence = padded_sequence.tolist()
#     print(padded_sequence)
    started = False
    word_seq = []
    for word in padded_sequence:
        if started:
            word_seq.append(word)
        else:
            if word != 0:
                started = True
                word_seq.append(word)
    
    sentences = list(map(sequence_to_text, [word_seq]))
    if len(sentences)>0:
        my_texts = []
        for word in sentences[0]:
            if word:
                my_texts.append(word)
            
        return ' '.join(my_texts)
    return None

def convert_y(y):
    result = []
    for index, col in enumerate(y_col):
        if index == y:
            result.append(1)
        else:
            result.append(0)
#     print(result)
    return result

def reshape_x_train(X_train_encoded_padded_words, r, c):
    x_train = []
    for row in X_train_encoded_padded_words:
        aa = np.array(row)

        aa = np.reshape(aa,(r ,c))
#         print(aa)
        x_train.append(aa)
    return np.array(x_train)
    


# # Wgan gp 

# In[11]:


from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np


# In[27]:



class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self, r,c ,vocal_size):
        self.img_rows = r
        self.img_cols = c
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator(4,4)
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self, r, c):
        i = r/4
        model = Sequential()

        model.add(Dense(128 * 1 * 1, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((1, 1, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X_train, epochs, batch_size, sample_interval=50):

        # Load the dataset
#         (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
#         X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        csvfile = 'wgan.csv'
        c = 6
        noise = np.random.normal(0, 1, (c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        # Rescale images 0 - 1

        gen_imgs = Tokenizer_vocab_size*gen_imgs
        
        
        int_arr = np.array(gen_imgs, dtype='int')
#         print(int_arr[0])
        
        
#         print(len(int_arr[0,:,:,0]))
#         fig, axs = plt.subplots(r, c)
        cnt = 0

        for j in range(c):
            sentence = padded_sequence_to_text(int_arr[cnt])
#             result = convert_y(sampled_labels[cnt])
            if len(sentence) <= 0:
                continue
            print(sentence)
            cnt += 1
            df = pd.read_csv(csvfile)# Loading a csv file with headers 
            data = {
                'sentence':sentence,
            }
#             for index, col in enumerate(y_col):
#                 data[col] = result[index]
            df = df.append(data, ignore_index=True)
            df.to_csv(csvfile, index = False,  encoding='utf-8')



# In[ ]:


r = 4
c = 4
x_train = reshape_x_train(X_train_encoded_padded_words, r, c)
# Rescale -1 to 1
x_train = (x_train.astype(np.float32) - Tokenizer_vocab_size/2) / (Tokenizer_vocab_size/2)
wgan = WGANGP(r,c,Tokenizer_vocab_size)
wgan.train(x_train, epochs=30000, batch_size=32, sample_interval=200)

