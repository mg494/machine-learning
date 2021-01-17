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
arguments and parameters
no positional arg: eval last trained model
"train": train new model

SOURCE: directory with subdirs for every category
SAVE_MODEL: path to directory for saved models
CATEGORIES: list of category numbers to train and test on, list of int
MODEL_SUFFIX: suffix added to output files (lossfunction and test eval)
"""

SOURCE = r"C:/Users/Marc/Documents/python_projects/machine_learning/thumbnails/"
SAVE_MODEL = r"C:\Users\Marc\Documents\python_projects\machine_learning\saved_models"
CATEGORIES = [17,2] # Pets vs. Cars [15,2] # Sport vs Cars [17,2]
MODEL_SUFFIX = "ncs_featurewise"

argsin = sys.argv[1:]
nargsin = len(argsin)

"""
prepare the data
"""

# pick samples from category subdirectories
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

# print out stats
min_samples = min(number_of_samples)
print("number of samples:",min_samples)

# loading image data and arranging to arrays
thumbnail_samples = []

y_category = []
number_of_categories=len(CATEGORIES)
for category,idx in zip(CATEGORIES,range(number_of_categories)):
	thumbnail_samples_per_category = []
	print("++++++",category,idx,"+++++")
	samples = videos_in_dir[idx]

	for thumbnail_file in samples[:min_samples]:
		thumbnail = cv2.imread(SOURCE+str(category)+r'/'+thumbnail_file) #Reading the thumbnail (OpenCV)

		"""
		# normalize, center samplewise and globally, standardize globally
		thumbnail_sample_norm = np.asarray(thumbnail)/255.0
		thumbnail_sample = (thumbnail_sample_norm - thumbnail_sample_norm.mean())/thumbnail_sample_norm.std()
		"""
		thumbnail_sample = np.asarray(thumbnail)/255.0

		# append to dataset
		thumbnail_samples_per_category.append(thumbnail_sample)

		truth_for_image = np.where(np.array(CATEGORIES)==category,1,0)
		y_category.append(truth_for_image)

	# center and standardize featurewise
	thumbnail_samples_per_category = np.asarray(thumbnail_samples_per_category)
	thumbnail_samples_per_category = (thumbnail_samples_per_category - thumbnail_samples_per_category.mean())/thumbnail_samples_per_category.std()
	thumbnail_samples.append(thumbnail_samples_per_category)

thumbnail_samples = np.concatenate(thumbnail_samples,axis=0)
# make input arrays
x = np.asarray(thumbnail_samples)
y = np.asarray(y_category)
print("shape of input array:",x.shape)
print("shape of output array:",y.shape)

#print(x[0].min(),x[0].max(),x[0].mean(),x[0].std())


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
	epochs = 50

	model.compile(optimizer=Optimizer.SGD(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
	model.summary()

	csv_logger = CSVLogger('./data/thumbnail_stats/thumbnail_training_'+MODEL_SUFFIX+'.log', separator=',', append=False)
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,callbacks=[csv_logger],validation_split=0.1)	# ,callbacks=[plot_losses]  plot_losses = PlotLosses()
	model.save(SAVE_MODEL+"/"+MODEL_SUFFIX)

else:
	model = keras.models.load_model(SAVE_MODEL+"/"+MODEL_SUFFIX)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
test_results = open("./data/thumbnail_stats/thumbnail_test_"+MODEL_SUFFIX+".txt","w+")
test_results.write("{}\t{}\n".format( score[0], score[1]))
test_results.close()

# print latest loss fcn
csv_history = pd.read_csv('./data/thumbnail_stats/thumbnail_training_'+MODEL_SUFFIX+'.log')
csv_history.set_index("epoch")
print(csv_history.columns)
csv_history.drop('epoch',axis=1,inplace=True)
csv_history.plot()#logy=True
plt.title('model performance')
plt.ylabel('quantity')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.savefig("./data/figures/mlp/lossfunction_"+MODEL_SUFFIX+".png")
plt.show()