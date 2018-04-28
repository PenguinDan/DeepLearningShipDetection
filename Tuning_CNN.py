# -*- coding: utf-8 -*-
from CNN import build_CNN, train_CNN, test_CNN
from Helper import load_data
import numpy as np
import Constants

def k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, structure, data, labels, batch_size, epoch):
    
    val_size = int(len(data)/k)
    all_loss_scores = []
    all_accuracy_scores = []
    
    for i in range(k):
        print('Fold #', i)
        val_data = data[i * val_size: (i+1) * val_size]
        val_labels = labels[i * val_size: (i+1) * val_size]
        model = build_CNN(learning_rate, decay_rate, momentum_value, structure)

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

        model = train_CNN(model, train_data, train_labels, batch_size, epoch)
        loss, accuracy = test_CNN(model, val_data, val_labels, batch_size)
        all_loss_scores.append(loss)
        all_accuracy_scores.append(accuracy)

    average_loss = np.mean(all_loss_scores)
    average_accuracy = np.mean(all_accuracy_scores)

    return model, average_loss, average_accuracy

def test_models(k, learning_rate, decay_rate, momentum_value, data, labels, batch_size, epoch):
    model_1, loss_1, accuracy_1 = k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, [False, True, True, True, True], data, labels, batch_size, epoch)
    model_2, loss_2, accuracy_2 = k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, [True, False, True, True, True], data, labels, batch_size, epoch)
    model_3, loss_3, accuracy_3 = k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, [True, True, False, True, True], data, labels, batch_size, epoch)
    model_4, loss_4, accuracy_4 = k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, [True, True, True, False, True], data, labels, batch_size, epoch)
    model_5, loss_5, accuracy_5 = k_fold_cross_validation(k, learning_rate, decay_rate, momentum_value, [True, True, True, True, False], data, labels, batch_size, epoch)

    print("\n\n\n\nModel 1 Loss: ", loss_1, "       Accuracy: ", accuracy_1)
    print("Model 2 Loss: ", loss_2, "       Accuracy: ", accuracy_2)
    print("Model 3 Loss: ", loss_3, "       Accuracy: ", accuracy_3)
    print("Model 4 Loss: ", loss_4, "       Accuracy: ", accuracy_4)
    print("Model 5 Loss: ", loss_5, "       Accuracy: ", accuracy_5)


x, y = load_data(Constants.TRAINING_SMALL_IMAGE_DATASET)
test_models(5, 0.0010, 1e-6, 0.9, x, y, 50, 100)



