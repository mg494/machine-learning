

# prerequisites
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from scipy.stats import norm
import cv2
from sklearn.model_selection import train_test_split
from skimage import color

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

dataset = "yt"

if dataset == "yt":
	# data load
	# parameters
	SOURCE = r"C:/Users/Marc/Documents/python_projects/machine_learning/thumbnails/"
	CATEGORIES = [10]

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

	        thumbnail = color.rgb2gray(thumbnail)
	        thumbnail_sample = np.reshape(np.asarray(thumbnail.astype("float32")/255.),10800)
	        #thumbnail_sample.resize((96,120,3))

	        # append to dataset
	        thumbnail_samples.append(thumbnail_sample)      #_per_category

	        #truth_for_image = np.where(np.array(CATEGORIES)==category,1,0)
	        y_category.append(category)

	# make input arrays
	x = np.asarray(thumbnail_samples)
	#y = np.asarray(y_category)
	print("shape of input array:",x.shape)
	#print("shape of output array:",y.shape)

	# split the data
	x_train, x_test = train_test_split(x, test_size=0.3, random_state=42)
	print(x_train.shape, x_test.shape)


else:
	# data load
	fashion_mnist = keras.datasets.fashion_mnist

	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	# (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
	x_train, x_test = train_images.astype('float32')/255., test_images.astype('float32')/255.
	x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
	print(x_train.shape, x_test.shape)


# network parameters
batch_size, n_epoch = 100, 50
n_hidden, z_dim = 256, 2

# encoder
x = Input(shape=(x_train.shape[1:]))
x_encoded = Dense(n_hidden, activation='relu')(x)
x_encoded = Dense(n_hidden//2, activation='relu')(x_encoded)
x_encoded = Dense(n_hidden//2, activation='relu')(x_encoded)


mu = Dense(z_dim)(x_encoded)
log_var = Dense(z_dim)(x_encoded)

# sampling function
def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps

z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])

z_decoder1 = Dense(n_hidden//2, activation='relu')
z_decoder2 = Dense(n_hidden, activation='relu')
z_decoder3 = Dense(n_hidden, activation='relu')
y_decoder = Dense(x_train.shape[1], activation='sigmoid')

z_decoded = z_decoder1(z)
z_decoded = z_decoder2(z_decoded)
z_decoded = z_decoder3(z_decoded)
y = y_decoder(z_decoded)

print(x_train.shape[1])

# loss
reconstruction_loss = objectives.binary_crossentropy(x, y) * x_train.shape[1]
kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)
vae_loss = reconstruction_loss + kl_loss

# build model
vae = Model(x, y)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

# train
vae.fit(x_train,
       shuffle=True,
       epochs=n_epoch,
       batch_size=batch_size,
       validation_data=(x_test, None), verbose=1)


# build encoder
encoder = Model(x, mu)
encoder.summary()

# Plot of the digit classes in the latent space
x_test_latent = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_latent[:, 0], x_test_latent[:, 1], c=test_labels)
plt.colorbar()
plt.show()

# build decoder
decoder_input = Input(shape=(z_dim,))
_z_decoded = z_decoder1(decoder_input)
_z_decoded = z_decoder2(_z_decoded)
_y = y_decoder(_z_decoded)
generator = Model(decoder_input, _y)
generator.summary()

# display a 2D manifold of the digits
n = 15 # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()



preview = vae.predict(x_train, batch_size=batch_size)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(preview[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()