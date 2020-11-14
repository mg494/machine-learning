"""
the goal is to predict the category of a video from a number of features such as
likes, dislikes, comment count and number of days in trends.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# package settings
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)

# read dataset
dataframe = pd.read_pickle("./data/videos.pkl")
print("total number of videos:", len(dataframe.index))
print("unique categories:\n",dataframe["category"].unique())
print("count of unique categories:\n",dataframe["category"].value_counts())

# choose features and category to predict
predict = "category"
prediction_value = 24
features = ["last_likes","last_views"] #,"no_of_entries","publish_time"

subset = dataframe[features + [predict]]
positive = subset[subset[predict]==prediction_value]
negative = subset[subset[predict]!=prediction_value]

if (nsamples := len(positive.index)) < 100:
	print("less than 100 samples")
	sys.exit(0)

print("number of positive samples:",nsamples)

# shuffle negative data
#negative.reset_index(inplace=True)
#negative.reindex(np.random.permutation(negative.index))
#negative = negative.iloc[range(nsamples)]
#negative.drop('index',axis=1,inplace=True)

# merge into one frame
modeldata = pd.DataFrame()
modeldata = modeldata.append(positive).dropna()
modeldata = modeldata.append(negative).dropna()

modeldata.reset_index(inplace=True)
modeldata.drop("index",axis=1,inplace=True)
print(modeldata.head())

if (number_of_features:=len(features)-1) > 2:
	# plots
	fig,ax = plt.subplots()
	plt.scatter(positive[features[0]],positive[features[1]],color="g")
	plt.scatter(negative[features[0]],negative[features[1]],color="r")

## preprocess model data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA

X = modeldata.drop(predict,axis=1).to_numpy().astype("float")
y = np.where(modeldata[predict].to_numpy()==prediction_value,1,0)

# scale data
min_max_scaler = preprocessing.MinMaxScaler()
Xs = min_max_scaler.fit_transform(X)

pca = PCA(n_components=number_of_features)
Xspca = pca.fit_transform(Xs)

def classifier(X,y):
	# split into test and training data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,random_state=0)
	# classifier and predict
	clf = SVC()
	clf.fit(X_train, y_train)
	# return prediction and score on the testdata
	return clf.predict(X_test), clf.score(X_test,y_test)

#ys_p,score_scaled = classifier(Xs,y)
#yspca_p,score_scaled_pca = classifier(Xspca,y)
y_p,score = classifier(X,y)
print("score:",score) #,score_scaled,score_scaled_pca

sys.exit()

failure_rates = []
for eps in (tolerances := np.logspace(1e-5,1,100)):
	# classifier and predict
	clf = LinearSVC(tol=eps)
	clf.fit(X_train, y_train)
	y_p = clf.predict(X_test)

	failure_rate = sum(abs(y_p-y_test))/len(y_test)
	failure_rates.append(failure_rate)

fig1,ax1 = plt.subplots()
ax1.scatter(range(len(y_test)),y_test,color="g",marker="+")
ax1.scatter(range(len(y_p)),y_p,color="k",marker="+")
fig1.savefig("./data/figures/svm.png")

fig2,ax2 = plt.subplots()
ax2.plot(tolerances,failure_rates)
fig2.savefig("./data/figures/svm_optim.png")