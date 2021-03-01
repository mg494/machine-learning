import keras, os ,sys, glob
from keras import layers
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# parameters
SOURCE = r"C:/Users/Marc/Documents/python_projects/machine_learning/thumbnails/"
CATEGORIES = [17,2]

# load dataset
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
		# samplewise
		# normalize, center samplewise and globally, standardize globally
		thumbnail_sample_norm = np.asarray(thumbnail)/255.0
		thumbnail_sample = (thumbnail_sample_norm - thumbnail_sample_norm.mean())/thumbnail_sample_norm.std()
		"""
		thumbnail_sample = np.asarray(thumbnail)/255.0
		thumbnail_sample.resize((96,120,3))

		# append to dataset
		thumbnail_samples.append(thumbnail_sample)		#_per_category

		truth_for_image = np.where(np.array(CATEGORIES)==category,1,0)
		y_category.append(truth_for_image)


# make input arrays
x = np.asarray(thumbnail_samples)
print("shape of input array:",x.shape)

# split the data
x_train, x_test = train_test_split(x, test_size=0.3, random_state=42)

# This is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# ENCODER
image_shape = (96,120,3)
input_img = layers.Input(shape=image_shape)
x = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
#x = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(x)
#x = layers.MaxPooling2D((2, 2), padding='same')(x)
#x = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(x)
#x = layers.MaxPooling2D((2, 2), padding='same')(x)
encoded = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)

latentSize = (12,15,32)

# DECODER
direct_input = layers.Input(shape=latentSize)
#x = layers.Conv2D(192, (1, 1), activation='relu', padding='same')(direct_input)
#x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(direct_input)
x = layers.UpSampling2D((2, 2))(x)
#x = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(x)
#x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# COMPILE
encoder = keras.models.Model(input_img, encoded)
decoder = keras.models.Model(direct_input, decoded)
autoencoder = keras.models.Model(input_img, decoder(encoded))
encoder.summary()
decoder.summary()
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')


autoencoder.fit(x_train, x_train,
                epochs=2,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
