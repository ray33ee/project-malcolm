from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras

import numpy as np

import tensorflow as tf

import os

import pickle

import math

import matplotlib.pyplot as plt

import json

model = None

data, labels = None, None

training_data, training_labels = None, None

validation_data, validation_labels = None, None

model_info = None

while True:

    input_string = input("> ")

    command = input_string[0]

    arguments = input_string[1:]

    if arguments:
        arguments = json.loads(arguments)

    if command == "D":
        path = arguments["path"]

        if not os.path.exists(path):
            print("Path not found.")
            quit(1)

        # If the model has already been initialised, make sure the data file is compatible with it
        with open(path, "rb") as file:
            data, labels = pickle.load(file)

    elif command == "S":
        total_count = data.__len__()
        ratio = int(arguments["ratio"] * total_count)

        training_data = data[:ratio]
        training_labels = labels[:ratio]

        validation_data = data[ratio:]
        validation_labels = labels[ratio:]

    elif command == "l":
        with open(arguments["path"], "rb") as file:
            model = pickle.load(file)
        # If the data has already been loaded, make sure the loaded model is compatible with it
    elif command == "s":
        with open(arguments["path"], "wb") as file:
            pickle.dump(model, file)
    elif command == "c":
        model = Sequential()

        model_info = arguments
    elif command == "L":
        if model.layers.__len__() == 0:
            # If this is the first layer, add the 'input_shape' argument
            print(data.shape[1])
            model.add(Dense(**arguments, input_shape=(data.shape[1],)))
        else:
            model.add(Dense(**arguments))
    elif command == "C":
        # Before we compile, add another final layer, the output layer
        if model_info["type"] == "regression": # If it's a regression model, we can infer the output size from the label shape
            model.add(Dense(**arguments["layer_args"], units=labels.shape[1]))
        elif model_info["type"] == "classification": # If it's a classification we need to be given the output size
            model.add(Dense(**arguments["layer_args"], units=model_info["class_count"]))
        model.compile(**arguments["compile_args"])
    elif command == "T":
        model.fit(training_data, training_labels, **arguments)
    elif command == "E":
        results = model.evaluate(validation_data, validation_labels)
        print(results)
    elif command == "G":
        # Since an input vector may have multiple entries, we ask the user to select a single scalar in vector to use as
        # independent variable.
        independent_index = arguments["independent_index"]
        dependent_index = arguments["dependent_index"]

        x_values = validation_data.transpose()[independent_index]
        y1_values = validation_labels.transpose()[dependent_index]

        y2_values = model.predict(validation_data).transpose()[dependent_index]

        plt.plot(x_values, y1_values, '.', x_values, y2_values, '.')
        plt.show()

    elif command == "q":
        break
