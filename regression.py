import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import math

pd.set_option('display.max_columns', None)

"""
exploratory analysis of acumulation of likes or views over time for videos in the trend
ultimativly the model be should be valid for infering the current likes or views

"""

# import numerical dataset
dataframe = pd.read_pickle("./data/videos.pkl")


# sort by number of data points
dataframe.sort_values("no_of_entries",ascending=False,inplace=True)

# print for overview
print(dataframe.head())

# select variables of the model
dependant = "likes"
independant = "views"
idx_video = 900

# init plot
fig, ax = plt.subplots()
plt.xlabel(independant)
plt.ylabel(dependant)
plt.title(dataframe["country"].iloc[idx_video])

x = dataframe[independant].to_numpy()[idx_video]
y = dataframe[dependant].to_numpy()[idx_video]

# plot measured data
ax.scatter(x,y, s=5)

# define model with statsmodels
X = sm.add_constant(x)
model = ols(formula="likes ~ views + np.sqrt(views) + np.log(views) ",data = pd.DataFrame(data={"likes":y,"views":x}))  #

# results
results = model.fit()

print(results.summary())
print(results.params)
# plot hypothesis
ax.plot(x,results.fittedvalues,color="r")

"""

test_sizes = [0.2,0.3,0.4,0.5]
for test_size in test_sizes:
	# select training data and randomize samples
	train_idx = np.random.rand(no_of_entries[0]) < test_size
	x = df_by_video[independant].to_numpy()[train_idx]
	y = df_by_video[dependant].to_numpy()[train_idx]

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
	ax.plot(x,results.fittedvalues,label=str(test_size))

	# show results
	print(results.summary())
"""
plt.legend(loc="best")
plt.savefig("./data/figures/regression_"+independant+"_"+dependant+".png")