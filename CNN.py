# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.optimizers import SGD
from keras import metrics
from keras.models import load_model 
import h5py

#Creates and returns a CNN model with a given architecture
#Structure is a boolean array of size 5, if a given index of structure is False it turns off the cooresponding pooling layer
def build_CNN(learning_rate, decay_rate, momentum_value, structure):
    
    #Build the CNN model by modifying VGGNet to fit the image size
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same', input_shape = (80, 80, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    if structure[0]:
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
    if structure[1]:
        model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
    if structure[2]:
        model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
    if structure[3]:
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
    if structure[4]:
        model.add(MaxPooling2D((2, 2)))

    #add a fully connected layer
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    #Set up the input size
    img_input = Input(shape=(80, 80, 3))
    output = model(img_input)
    
    #Set up optimizer along with the loss function
    sgd = SGD(lr = learning_rate, decay = decay_rate, momentum = momentum_value, nesterov = True)
    model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=[metrics.binary_accuracy])
    
    return model

#Trains a given model with given data and labels
# Batch is the number of samples per gradient update
#Epoch size is how many epochs to train the model
def train_CNN(model, data, labels, batch, epoch_size):
    
    model.fit(data, labels, batch_size=batch, epochs=epoch_size)
    
    return model

#Trains a given model using a generator object (which augments the data)
#Steps defines how many batches of augmented data the model should be trained on per epoch
#Epoch size is how many epochs to train the model
def train_generator_CNN(model, generator, steps, epoch_size):
    
    model.fit_generator(generator, steps_per_epoch=steps, epochs=epoch_size)
    
    return model

#Tests the given model on the data and labels on a given batch size
def test_CNN(model, data, labels, batch):
    
    loss, accuracy = model.evaluate(data, labels, batch_size=batch)
    print("Loss: ", loss, "     Accuracy: ", accuracy)
    
    return loss, accuracy

#Saves the model passed in to the given file location
def save_CNN(model, location):
    model.save(location)

#Loads and returns a trained model from the given file location
def load_CNN(location):
    return load_model(location)

#Takes in a trained CNN model and data and outputs predicted values for every datapoint 
def predict_CNN(model, x):
    y = model.predict(x, 50)
    return y