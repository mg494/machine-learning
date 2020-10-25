import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

"""
exploratory analysis of acumulation of likes or views over time for videos in the trend
ultimativly the model be should be valid for infering the current likes or views

"""

# import numerical dataset
dataframe = pd.read_pickle("./data/dataset.pkl")

# select single country from time series
selected_country = "GB"
dependant = "likes"
independant = "days"

# filter frame
df_by_country = dataframe[dataframe.country==selected_country]

# number of unique videos
unique_videos = np.array(df_by_country.sort_values(dependant,ascending=False)["video_id"].unique())
no_of_videos = len(unique_videos)

# init plot
fig, ax = plt.subplots()

for video in unique_videos[0:6]:
	df_by_video = df_by_country[df_by_country["video_id"] == video]

	# reset days
	df_by_video["days"] = (df_by_video["days"] - df_by_video["days"].min()).add(1)

	# select data and plot it
	ax.scatter(df_by_video[independant],df_by_video[dependant],marker="+")

	# select training data ?
	x = df_by_video[independant].to_numpy()
	y = df_by_video[dependant].to_numpy()

	# transform to logarithmic regression
	x = np.log(x)

	# train model
	X = sm.add_constant(x)
	model = sm.OLS(y,X)

	# get the results
	results = model.fit()

	# transform to linear scale
	x = np.exp(x)

	# plot OLS
	ax.plot(x,results.fittedvalues)

	# show results
	print(results.summary())

plt.savefig("./data/figures/test.png")