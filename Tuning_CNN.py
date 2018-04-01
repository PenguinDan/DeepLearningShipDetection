# -*- coding: utf-8 -*-
from CNN import *
from Helper import load_data
import Constants

x, y = load_data(Constants.TRAINING_SMALL_IMAGE_DATASET)
model = build_CNN(0.001, 1e-6, 0.9)
model = train_CNN(model, x[500:], y[500:], 50, 20)
test_CNN(model, x[:500], y[:500], 50)