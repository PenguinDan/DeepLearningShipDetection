# -*- coding: utf-8 -*-
from CNN import build_CNN, train_CNN, test_CNN, save_CNN, load_CNN, train_generator_CNN, load_pretrained_VGG16, data_augmentation
from Helper import load_data
import numpy as np
import Constants

def k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, structure, data, labels, batch_size, epoch):
    
    #initialize values
    val_size = int(len(data)/k)
    all_loss_scores = []
    all_accuracy_scores = []
    
    #run through k iterations
    for i in range(k):
        print('Fold #', i)

        #Divide the data into training and validation
        #Create the validation data
        val_data = data[i * val_size: (i+1) * val_size]
        val_labels = labels[i * val_size: (i+1) * val_size]
        model = build_CNN(learning_rate, decay_rate, momentum_value, structure)

        #Create the training dataset
        train_data = np.concatenate(
            [data[:i * val_size],
            data[(i+1) * val_size:]], 
            axis = 0
        )

        train_labels = np.concatenate(
            [labels[:i * val_size],
            labels[(i+1) * val_size:]], 
            axis = 0
        )

        #Train the model
        model = train_CNN(model, train_data, train_labels, batch_size, epoch)

        #test the model on the validation dataset
        loss, accuracy = test_CNN(model, val_data, val_labels, batch_size)

        #save the loss and accuracy values
        all_loss_scores.append(loss)
        all_accuracy_scores.append(accuracy)

    #average all the loss and accuracy values from all k folds
    average_loss = np.mean(all_loss_scores)
    average_accuracy = np.mean(all_accuracy_scores)

    #return the averages
    return model, average_loss, average_accuracy


def k_fold_cross_validation_with_generator(k, learning_rate, decay_rate, momentum_value, structure, data, labels, steps, epoch):
    
    #initialize values
    val_size = int(len(data)/k)
    all_loss_scores = []
    all_accuracy_scores = []
    
    #run through k iterations
    for i in range(0,k):
        run = 'Fold #'+str(i)
        print(run)

        #Divide the data into training and validation
        #Create the validation data
        val_data = data[i * val_size: (i+1) * val_size]
        val_labels = labels[i * val_size: (i+1) * val_size]
        model = build_CNN(learning_rate, decay_rate, momentum_value, structure)

        #Create the training dataset
        train_data = np.concatenate(
            [data[:i * val_size],
            data[(i+1) * val_size:]], 
            axis = 0
        )

        train_labels = np.concatenate(
            [labels[:i * val_size],
            labels[(i+1) * val_size:]], 
            axis = 0
        )

        #Create generator
        generator = data_augmentation(train_data, train_labels)

        #Train the model
        model = train_generator_CNN(model, generator, steps, epoch, (val_data, val_labels), run)

        #test the model on the validation dataset
        loss, accuracy = test_CNN(model, val_data, val_labels, 50)

        #save the loss and accuracy values
        all_loss_scores.append(loss)
        all_accuracy_scores.append(accuracy)

    #average all the loss and accuracy values from all k folds
    average_loss = np.mean(all_loss_scores)
    average_accuracy = np.mean(all_accuracy_scores)

    #return the averages
    return model, average_loss, average_accuracy

#Tests the 5 different possible variations of VGG-16 for our data using k-fold validation and outputs the results to the console
def test_models(k, learning_rate, decay_rate, momentum_value, data, labels, batch_size, epoch):
    model_1, loss_1, accuracy_1 = k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, [False, True, True, True, True], data, labels, batch_size, epoch)
    model_2, loss_2, accuracy_2 = k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, [True, False, True, True, True], data, labels, batch_size, epoch)
    model_3, loss_3, accuracy_3 = k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, [True, True, False, True, True], data, labels, batch_size, epoch)
    model_4, loss_4, accuracy_4 = k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, [True, True, True, False, True], data, labels, batch_size, epoch)
    model_5, loss_5, accuracy_5 = k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, [True, True, True, True, False], data, labels, batch_size, epoch)

    #output results to the console
    print("\n\n\n\nModel 1 Loss: ", loss_1, "       Accuracy: ", accuracy_1)
    print("Model 2 Loss: ", loss_2, "       Accuracy: ", accuracy_2)
    print("Model 3 Loss: ", loss_3, "       Accuracy: ", accuracy_3)
    print("Model 4 Loss: ", loss_4, "       Accuracy: ", accuracy_4)
    print("Model 5 Loss: ", loss_5, "       Accuracy: ", accuracy_5)


x, y = load_data(Constants.TRAINING_SMALL_IMAGE_DATASET, (224,224))


# model_1, loss_1, accuracy_1 = k_fold_cross_validation_with_generator(5, 0.0010, 1e-6, 0.9, [False, True, True, True, True], x, y, 200, 100)

model = load_pretrained_VGG16(0.0001, 1e-6, 0.75, (224,224,3))
generator = data_augmentation(x[500:], y[500:], 25)
run = "pretrained_attempt_6"
model = train_generator_CNN(model, generator, 400, 30, (x[0:500], y[0:500]), run)
# train_CNN(model, x[500:], y[500:], 50, 30)
save_CNN(model, Constants.PRETRAINED_CNN_MODEL_SIX)

# model = load_CNN("pretrained_attempt_2.h5")
# test_CNN(model, x[:500], y[:500], 50)


