from keras.datasets import cifar10

(train_X, train_Y), (test_X, test_Y) = cifar10.load_data()

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt


print('Training data shape : ', train_X.shape, train_Y.shape)

print('Testing data shape : ', test_X.shape, test_Y.shape)

classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

plt.figure(figsize=[5,5])
# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

plt.show()

shape_range = len(np.unique(train_X))
print(shape_range) 

train_X = train_X.reshape(-1, 32,32,3)
test_X = test_X.reshape(-1, 32,32,3)

#Configuration for hot key
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

print('Original label:', train_Y[2])
print('After conversion to one-hot:', train_Y_one_hot[2])


from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

train_X.shape,valid_X.shape,train_label.shape,valid_label.shape

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 25
num_classes = 10

cifa_model = Sequential()
cifa_model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(32,32,3),padding='same'))
cifa_model.add(LeakyReLU(alpha=0.1))
cifa_model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(32,32,3),padding='same'))
cifa_model.add(LeakyReLU(alpha=0.1))
cifa_model.add(MaxPooling2D((2, 2),padding='same'))
# cifa_model.add(Dropout(0.25))    
cifa_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
cifa_model.add(LeakyReLU(alpha=0.1))
cifa_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
cifa_model.add(Dropout(0.25))    
cifa_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
cifa_model.add(LeakyReLU(alpha=0.1))                    
cifa_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# cifa_model.add(Dropout(0.4))    
cifa_model.add(Flatten())
cifa_model.add(Dense(128, activation='linear'))
cifa_model.add(LeakyReLU(alpha=0.1))          
cifa_model.add(Dropout(0.3))        
cifa_model.add(Dense(num_classes, activation='softmax'))

cifa_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

cifa_model.summary()

cifa_train = cifa_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

test_eval = cifa_model.evaluate(test_X, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

cifa_model.save('model_file.h5')

accuracy = cifa_train.history['acc']
val_accuracy = cifa_train.history['val_acc']
loss = cifa_train.history['loss']
val_loss = cifa_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
