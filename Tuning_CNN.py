# -*- coding: utf-8 -*-
from CNN import build_CNN, train_CNN, test_CNN
from Helper import load_data
import numpy as np
import Constants

def k_fold_cross_validation(k, model, data, labels, batch_size, epoch):
    
    val_size = len(data)/k
    all_loss_scores = []
    all_accuracy_scores = []
    
    for i in range(k):
        print('Fold #', i)
        val_data = data[i * val_size: (i+1) * val_size]
        val_labels = labels[i * val_size: (i+1) * val_size]

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


x, y = load_data(Constants.TRAINING_SMALL_IMAGE_DATASET)
model_1 = build_CNN(0.001, 1e-6, 0.9, [False, True, True, True, True])
model_2 = build_CNN(0.001, 1e-6, 0.9, [True, False, True, True, True])
model_3 = build_CNN(0.001, 1e-6, 0.9, [True, True, False, True, True])
model_4 = build_CNN(0.001, 1e-6, 0.9, [True, True, True, False, True])
model_5 = build_CNN(0.001, 1e-6, 0.9, [True, True, True, True, False])

model_1, loss_1, accuracy_1 = k_fold_cross_validation(5, model_1, x, y, 50, 20)
model_2, loss_2, accuracy_2 = k_fold_cross_validation(5, model_2, x, y, 50, 20)
model_3, loss_3, accuracy_3 = k_fold_cross_validation(5, model_3, x, y, 50, 20)
model_4, loss_4, accuracy_4 = k_fold_cross_validation(5, model_4, x, y, 50, 20)
model_5, loss_5, accuracy_5 = k_fold_cross_validation(5, model_5, x, y, 50, 20)

print("Model 1 Loss: ", loss_1, "       Accuracy: ", accuracy_1)
print("Model 2 Loss: ", loss_2, "       Accuracy: ", accuracy_2)
print("Model 3 Loss: ", loss_3, "       Accuracy: ", accuracy_3)
print("Model 4 Loss: ", loss_4, "       Accuracy: ", accuracy_4)
print("Model 5 Loss: ", loss_5, "       Accuracy: ", accuracy_5)
