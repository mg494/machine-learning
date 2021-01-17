import keras
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import CSVLogger
import pandas as pd
import numpy as np
import os,sys
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""
# get input arguments
no input: eval last trained model
train: train new model
"""

argsin = sys.argv[1:]
nargsin = len(argsin)

"""
prepare the data
"""

SOURCE = r"C:/Users/Marc/Documents/python_projects/machine_learning/thumbnails/"
SAVE_MODEL = r"C:\Users\Marc\Documents\python_projects\machine_learning\saved_models"
CATEGORIES = [17,2] # Pets vs. Cars [15,2] # Sport vs Cars [17,2]
number_of_categories=len(CATEGORIES)

# pick samples from category folders
number_of_samples = []
videos_in_dir = []
for directory in CATEGORIES:
	path = SOURCE+str(directory)
	files_in_dir = os.listdir(path)

	# remove windows settings file from dataset
	try:
		files_in_dir.remove("Thumbs.db")
	except:
		pass

	number_of_samples.append(len(files_in_dir))
	videos_in_dir.append(files_in_dir)

min_samples = min(number_of_samples)
print("number of samples:",min_samples)

# preparing images
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

if nargsin > 0 and argsin[0] == "train":

	"""
	## Build the model
	"""
	model = keras.models.Sequential()

	# source: mnist inspired
	model.add(keras.Input(shape=(90,120,3)))
	model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(number_of_categories, activation="softmax"))

	"""
	## Train the model
	"""
	# number of images passed trough the network before parameter update
	batch_size = 64

	# number of times complete samples are passed trough the network
	epochs = 40

	#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
	model.compile(optimizer=Optimizer.SGD(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
	model.summary()

	csv_logger = CSVLogger('./data/thumbnail_stats/thumbnail_training.log', separator=',', append=False)
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,callbacks=[csv_logger],validation_split=0.1)	# ,callbacks=[plot_losses]  plot_losses = PlotLosses()
	model.save(SAVE_MODEL)

else:
	model = keras.models.load_model(SAVE_MODEL)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
test_results = open("./data/thumbnail_stats/thumbnail_test.txt","w+")
test_results.write("{}\t{}\n".format( score[0], score[1]))
test_results.close()

# print latest loss fcn
csv_history = pd.read_csv('./data/thumbnail_stats/thumbnail_training.log')
csv_history.set_index("epoch")
print(csv_history.columns)
csv_history.drop('epoch',axis=1,inplace=True)
csv_history.plot(logy=True)
plt.title('model performance')
plt.ylabel('quantity')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig("./data/figures/mlp/lossfunction.png")
plt.show()