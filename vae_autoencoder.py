import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist

import cv2
from sklearn.model_selection import train_test_split
from skimage import color
from PIL import Image

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

original_dim = 10800
intermediate_dim = 2048
latent_dim = 2
batch_size = 1000
epochs = 200
epsilon_std = 1.0

# data loader
dataset = "yt"

if dataset == "yt":
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
            thumbnail_sample = np.reshape(np.asarray(thumbnail.astype("float32")/255.),original_dim)
            #thumbnail_sample.resize((96,120,3))

            # append to dataset
            thumbnail_samples.append(thumbnail_sample)      #_per_category

            #truth_for_image = np.where(np.array(CATEGORIES)==category,1,0)
            y_category.append(category)

    # make input arrays
    x = np.asarray(thumbnail_samples)
    y = np.asarray(y_category)
    print("shape of input array:",x.shape)
    print("shape of output array:",y.shape)

elif dataset == "dogs":

    SOURCE = r"C:/Users/Marc/Documents/python_projects/machine_learning/all-dogs/"
    image_samples = []

    # CREATE RANDOMLY CROPPED IMAGES
    for i in range(500000):
        img = Image.open(SOURCE + IMAGES[i%len(IMAGES)])
        img = img.resize(( 100,int(img.size[1]/(img.size[0]/100) )), Image.ANTIALIAS)
        w = img.size[0]; h = img.size[1]; a=0; b=0
        if w>64: a = np.random.randint(0,w-64)
        if h>64: b = np.random.randint(0,h-64)
        img = img.crop((a, b, 64+a, 64+b))
        img.save('../cropped/'+str(i)+'.png','PNG')
        if i%100000==0: print('created',i,'cropped images')
        print('created 500000 cropped images')

    for file in os.listdir(SOURCE):

        img = img.resize(( 100,int(img.size[1]/(img.size[0]/100) )), Image.ANTIALIAS)
        w = img.size[0]; h = img.size[1]; a=0; b=0
        if w>64: a = np.random.randint(0,w-64)
        if h>64: b = np.random.randint(0,h-64)
        img = img.crop((a, b, 64+a, 64+b))
        img.save('../cropped/'+str(i)+'.png','PNG')

        image = cv2.imread(SOURCE+file) #Reading the image (OpenCV)
        image_sample = np.reshape(np.asarray(image.astype("float32")/255.),original_dim)

        # append to dataset
        image_samples.append(image_sample)      #_per_category

    x = np.asarray(thumbnail_samples)
    print("shape of input array:",x.shape)


# split the data
x_train, x_test,y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
print(x_train.shape, x_test.shape)


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(original_dim, activation='sigmoid')
])
print(latent_dim)
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)
print(x_pred.shape)
vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='rmsprop', loss=nll)

vae.fit(x_train,x_train,shuffle=True,epochs=epochs,batch_size=batch_size,validation_data=(x_test, x_test))

encoder = Model(x, z_mu)

# display a 2D plot of the digit classes in the latent space
z_test = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test,
            alpha=.4, s=3**2, cmap='viridis')
plt.colorbar()

# display a 2D manifold of the digits
n = 10  # figure with 15x15 digits

# linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z, since the prior of the latent space
# is Gaussian
u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))
z_grid = norm.ppf(u_grid)
x_decoded = decoder.predict(z_grid.reshape(n*n, 2))
x_decoded = x_decoded.reshape(n, n, 90, 120)

plt.figure(figsize=(10, 10))
plt.imshow(np.block(list(map(list, x_decoded))), cmap='gray')
plt.show()