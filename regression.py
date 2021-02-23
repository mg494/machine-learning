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

# drop video with too less datapoints
min_entries = 20
dataframe = dataframe[dataframe["no_of_entries"]>=min_entries]

# bugs bugs bugs - i dont know what happend - no bugs if IN is excluded
#dataframe.drop([134,1879,1878],axis=0,inplace=True)
print("no of entries: ", len(dataframe.index) )

# select variables of the model
dependant = "likes"
independant = "views"

# init plot
fig, ax = plt.subplots()
plt.xlabel(independant)
plt.ylabel(dependant)
#plt.title(dataframe["country"].iloc[idx_video])

plot_index = 3
count = 0
# init dataframe to store coefficients
parameters = []
for x,y in zip(dataframe[independant],dataframe[dependant]):

	# define model, a constant will be defined automatically as intercept
	formula_string = "{0} ~ {1} + np.sqrt({1}) + np.log({1})".format(dependant,independant)
	model = ols(formula=formula_string,data = pd.DataFrame(data={dependant:y,independant:x}))  #

	# results
	results = model.fit()
	parameters.append(np.array(results.params))

	if count == plot_index:
		# mean square error
		print(results.summary())

		# plot measured data
		ax.scatter(x,y, s=5)

		# plot hypothesis
		ax.plot(x,results.fittedvalues,color="r")
	count += 1


# overwrite array with DataFrame
column_name = independant+"_vs_"+dependant
parameters = pd.DataFrame(data={"video_id":dataframe["video_id"],column_name:parameters})
parameters.to_pickle("./data/regression_parameters.pkl")

plt.savefig("./data/figures/regression_"+independant+"_"+dependant+".png")

# multi dimensional
independant = ["views","likes"]
dependant = "comments"

formula_string = dependant +"~"
for idx,var in enumerate(independant):
	print(idx,var)
	if idx > 0:
		formula_string += "+ {0} + np.sqrt({0}) + np.log({0})".format(var)
	else:
		formula_string += "{0} + np.sqrt({0}) + np.log({0})".format(var)

print(formula_string)

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
#plt.legend(loc="best")