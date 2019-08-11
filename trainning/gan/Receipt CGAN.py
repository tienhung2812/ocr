#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division


# In[472]:



# print(X_train[0].shape)


# # Get receipt dataset

# In[1]:


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
train_df = pd.read_csv('../text_classification/31-07-vigroupped.csv',   encoding='utf-8')

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

maxWordCount= 10
maxDictionary_size=Tokenizer_vocab_size
X_train_encoded_padded_words = sequence.pad_sequences(X_train_encoded_words, maxlen=maxWordCount)


# In[2]:


X_train_encoded_padded_words.shape
# Y_train


# In[3]:


tokenizer.word_index['ngừ']


# In[4]:


# Convert Y_train
# range(0,len(y_col))[5]
y_train = []
for row in Y_train:
    for index,col in enumerate(range(0,len(y_col))):
        if row[col] == 1:
            y_train.append(index)
print(len(y_train))


# # Gan 

# In[5]:


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np


# In[6]:


def build_generator(img_shape, latent_dim,num_classes):

        model = Sequential()

        model.add(Dense(Tokenizer_vocab_size, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(Tokenizer_vocab_size*2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(Tokenizer_vocab_size*4))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(img_shape), activation='tanh'))
        model.add(Reshape(img_shape))

        model.summary()

        noise = Input(shape=(latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)


# In[7]:


def build_discriminator(img_shape, num_classes):

        model = Sequential()

        model.add(Dense(Tokenizer_vocab_size*2, input_dim=np.prod(img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(Tokenizer_vocab_size*2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(Tokenizer_vocab_size*2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)


# In[8]:


# Setting variable

img_rows = 1
img_cols = 10
channels = 1
img_shape = (img_rows, img_cols, channels)
num_classes = len(y_col)
latent_dim = 100

optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator(img_shape,num_classes)
discriminator.compile(loss=['binary_crossentropy'],
    optimizer=optimizer,
    metrics=['accuracy'])

# Build the generator
generator = build_generator(img_shape,latent_dim,num_classes)

# The generator takes noise and the target label as input
# and generates the corresponding digit of that label
noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))
img = generator([noise, label])

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated image as input and determines validity
# and the label of that image
valid = discriminator([img, label])

# The combined model  (stacked generator and discriminator)
# Trains generator to fool discriminator
combined = Model([noise, label], valid)
combined.compile(loss=['binary_crossentropy'],
    optimizer=optimizer)


# In[9]:


# Save model function

def save_model(generator,discriminator):
    def save(model, model_name):
        model_path = "saved_model/%s.json" % model_name
        weights_path = "saved_model/%s_weights.hdf5" % model_name
        options = {"file_arch": model_path,
                    "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    save(generator, "generator")
    save(discriminator, "discriminator")


# In[10]:


print(X_train_encoded_words[0])
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

# Creating texts 


# In[11]:


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
# print(X_train_encoded_padded_words[0])
# print(padded_sequence_to_text(X_train_encoded_padded_words[0]))


# In[12]:


def convert_y(y):
    result = []
    for index, col in enumerate(y_col):
        if index == y:
            result.append(1)
        else:
            result.append(0)
#     print(result)
    return result


# In[17]:


# sample_

def sample_images(epoch, generator):
        csvfile = 'cgan.csv'
        c = len(y_col)
        noise = np.random.normal(0, 1, (c, 100))
        sampled_labels = np.arange(0, len(y_col)).reshape(-1, 1)

        gen_imgs = generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
#         print(sampled_labels)
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = Tokenizer_vocab_size*gen_imgs
        
        
        int_arr = np.array(gen_imgs, dtype='int')
#         print(int_arr[0])
        
        
#         print(len(int_arr[0,:,:,0]))
#         fig, axs = plt.subplots(r, c)
        cnt = 0

        for j in range(c):
            sentence = padded_sequence_to_text(int_arr[cnt])
            result = convert_y(sampled_labels[cnt])
            if len(sentence) <= 0:
                continue
            print(sentence,':',sampled_labels[cnt])
            cnt += 1
            df = pd.read_csv(csvfile)# Loading a csv file with headers 
            data = {
                'sentence':sentence,
            }
            for index, col in enumerate(y_col):
                data[col] = result[index]
            df = df.append(data, ignore_index=True)
            df.to_csv(csvfile, index = False,  encoding='utf-8')
#                 axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
#                 axs[i,j].axis('off')
#                 cnt += 1
#         fig.savefig("images/%d.png" % epoch)
#         plt.close()


# In[486]:


X_train_encoded_padded_words


# In[14]:


x_train = []
for row in X_train_encoded_padded_words:
    aa = np.array(row)
    
    aa = np.reshape(aa,(1,10))
#     print(aa)
    x_train.append(aa)


# In[15]:


x_train = np.array(x_train)
x_train.shape


# In[19]:


# Traing
epochs = 20001
batch_size=32
sample_interval=200

# Load the dataset
# Load the dataset
# (X_train, y_train), (_, _) = mnist.load_data()
X_train = x_train
y_train = np.array(y_train)

# Configure input
half_vocab = Tokenizer_vocab_size/2
X_train = (X_train.astype(np.float32) - half_vocab) / half_vocab
X_train = np.expand_dims(X_train, axis=3)
y_train = y_train.reshape(-1, 1)

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random half batch of images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs, labels = X_train[idx], y_train[idx]

    # Sample noise as generator input
    noise = np.random.normal(0, 1, (batch_size, 100))

    # Generate a half batch of new images
    gen_imgs = generator.predict([noise, labels])

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
    d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    # Condition on labels
    sampled_labels = np.random.randint(0, len(y_col), batch_size).reshape(-1, 1)

    # Train the generator
    g_loss = combined.train_on_batch([noise, sampled_labels], valid)

    # Plot the progress
    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    # If at save interval => save generated image samples
    if epoch % sample_interval == 0:
        sample_images(epoch, generator)
        save_model(generator,discriminator)

