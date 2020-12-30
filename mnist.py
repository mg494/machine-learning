"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_classes = 10
input_shape = (28, 28, 1)


"""
## Prepare the data
"""
def mnist_param(network_topo=[
                            keras.Input(shape=input_shape),
                            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                            layers.MaxPooling2D(pool_size=(2, 2)),
                            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                            layers.MaxPooling2D(pool_size=(2, 2)),
                            layers.Flatten(),
                            layers.Dropout(0.5),
                            layers.Dense(num_classes, activation="softmax"),
                            ]):



    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")


    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    """
    ## Build the model
    """
    layer_string = "2Conv_relu"
    model = keras.Sequential(
        network_topo
    )

    model.summary()

    """
    ## Train the model
    """

    batch_size = 128
    epochs = 7

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    score = model.evaluate(x_test, y_test, verbose=0)
    return history, score


# Model / data parameters

names = ["relu","softmax", "sigmoid","tanh"]
topos = [   [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
            ],
            [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="softmax"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="softmax"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
            ],
            [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="sigmoid"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="sigmoid"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
            ],
            [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="tanh"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="tanh"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
            ]
            ]

fig,ax = plt.subplots(2)
ax[0].set_ylabel("loss")
ax[1].set_ylabel("accuracy")

train_results = pd.DataFrame()
test_results = pd.DataFrame()

test_results = open("./data/mnist_test_results.txt","w+")


for topo,function_name in zip(topos,names):
    # train model
    history,score = mnist_param(topo)

    ## Evaluate the trained model
    print(train_loss := history.history['loss'])
    train_accuracy = history.history['accuracy']

    test_loss=score[0]
    test_accuracy=score[1]

    test_results.write("{}\t{}\t{}\n".format(function_name,test_loss,test_accuracy))

    # store to dataframe
    train_results["loss_"+function_name] = train_loss
    train_results["accuracy_"+function_name] = train_accuracy

    ax[0].plot(train_loss,label=function_name)
    ax[1].plot(train_accuracy,label=function_name)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

test_results.close()
train_results.to_csv("./data/mnist_train_results.csv")
ax[0].legend(loc='upper right')
ax[1].legend(loc='lower right')
ax[0].set_title('training performance')
ax[1].set_xlabel('epoch')
fig.savefig("./data/figures/mnist_train_performance.png")

