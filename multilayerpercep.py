import keras
import numpy as np
import os,sys
import cv2
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
prepare the data
"""

SOURCE = r"C:/Users/Marc/Documents/python_projects/machine_learning/thumbnails/"
CATEGORIES = [22,25,17]
number_of_categories=len(CATEGORIES)

# pick samples from category folders
number_of_samples = []
videos_in_dir = []
for directory in CATEGORIES:
	path = SOURCE+str(directory)
	files_in_dir = os.listdir(path)

	number_of_samples.append(len(files_in_dir))
	videos_in_dir.append(files_in_dir)

min_samples = min(number_of_samples)
print("number of samples:",min_samples)

thumbnail_samples = []
y_category = []
for category,idx in zip(CATEGORIES,range(number_of_categories)):
	print("++++++",category,idx,"+++++")
	samples = videos_in_dir[idx]

	for thumbnail_file in samples[:min_samples]:
		thumbnail = cv2.imread(SOURCE+str(category)+r'/'+thumbnail_file) #Reading the thumbnail (OpenCV)
		thumbnail_samples.append(np.asarray(thumbnail))
		truth_for_image = np.where(np.array(CATEGORIES)==category,1,0)
		y_category.append(truth_for_image)

# make input arrays
x = np.asarray(thumbnail_samples)
y = np.asarray(y_category)
print("shape of input array:",x.shape)
print("shape of output array:",y.shape)

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(90,120,3)))
model.add(keras.layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(5,5))
model.add(keras.layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(keras.layers.Conv2D(140,kernel_size=(3,3),activation='relu'))
model.add(keras.layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(keras.layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(5,5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(180,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(50,activation='relu'))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(2,activation='softmax'))

"""
## Train the model
"""
batch_size = 128
epochs = 20

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

