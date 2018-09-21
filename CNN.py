# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.optimizers import SGD
from keras import metrics
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import h5py
import json

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
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
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
def train_generator_CNN(model, generator, steps, epoch_size, validation, run):
    
    history = model.fit_generator(generator, steps_per_epoch=steps, epochs=epoch_size, validation_data=validation).history

    #Reformat history
    pretty_history = ""
    for i in range(len(history["binary_accuracy"])):
        pretty_history_format = "Epoch #{4:<3}: Training Accuracy: {0:.16f}  Training Loss: {1:.16f}    Epoch #{5:<3}: Validation Accuracy: {2:.16f}  Validation Loss: {3:.16f}"
        pretty_history +=  pretty_history_format.format(history["binary_accuracy"][i], history["loss"][i], history["val_binary_accuracy"][i], history["val_loss"][i], str(i+1), str(i+1)) + "\n"
        if i%5 == 4:
            pretty_history +=  "\n"

    #Log file save location
    file_location = "text_logs" + "\\" + run + ".txt"
    print(file_location)
    
    #Save a text file with training information
    log_file = open(file_location, 'w')
    log_file.write(pretty_history)
    log_file.close()

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

#Loads and freezes the weights of VGG16's convolutional layers and adds trainable fully connected layers
def load_pretrained_VGG16(learning_rate, decay_rate, momentum_value, input_shape):
    #Get vgg16 weights
    vgg16_model = VGG16(weights="imagenet", include_top=False)

    #Freeze all convolutional layers
    # for layer in vgg16_model.layers:
    #     layer.trainable = False

    #Choose input format
    input = Input(shape=input_shape, name="image_input")

    #Use generated model
    vgg16_output = vgg16_model(input)

    #Add fully connected layers
    x = Flatten(name ='flatten')(vgg16_output)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='predications')(x)

    #Create the model
    sgd = SGD(lr = learning_rate, decay = decay_rate, momentum = momentum_value, nesterov = True)
    pretrained_model = Model(inputs=input, outputs=x)
    pretrained_model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=[metrics.binary_accuracy])

    return pretrained_model

#Returns an object that augments the dataset during training       
def data_augmentation(x, y, batches=50):
    datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range = 0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    return datagen.flow(x, y, batch_size=batches)