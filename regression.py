import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import mse
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
print(dataframe.head())
# bugs bugs bugs - i dont know what happend - no bugs if indian data is excluded
exclude = [134,1879,1878]
for idx in exclude:
	try:
		dataframe.drop(idx,axis=0,inplace=True)
	except KeyError:
		pass

print("no of entries: ", len(dataframe.index) )

"""
------- one dimensional model ------------
"""
# select variables of the model
dependant = "likes"
independant = "views"

# init plot
fig, ax = plt.subplots()
plt.xlabel(independant)
plt.ylabel(dependant)


# store fitted coefficients
parameters = []

# loop over all entries of the dataframe
# plot_index chooses the plot to save
plot_index = 0
count = 0
for x,y in zip(dataframe[independant],dataframe[dependant]):

	# test size
	x_train, x_test,y_train, y_test = train_test_split(x,y, test_size=0.5, random_state=42, shuffle=True)

	# define model, a constant will be defined automatically as intercept
	formula_string = "{0} ~ {1} + np.sqrt({1}) + np.log({1})".format(dependant,independant)
	model = ols(formula=formula_string,data = pd.DataFrame(data={dependant:y_train,independant:x_train}))  #

	# results
	results = model.fit()
	parameters.append(np.array(results.params))

	if count == plot_index:
		# output training data
		print("train shape:",x_train.shape)
		print("test shape:",x_test.shape)

		# print summary
		print(results.summary())

		# training error
		print("trainig mse", mse(y_train,results.fittedvalues))

		# test error
		y_pred = results.predict(pd.DataFrame(data={dependant:y_test,independant:x_test}))
		print("test mse",mse(y_pred,y_test))

		# plot measured data
		ax.scatter(x_train,y_train, s=5,label="training data")

		# plot hypothesis
		sort_idx = np.argsort(x_train)
		x_train = x_train[sort_idx]
		y_fit = results.fittedvalues[sort_idx]

		x_hypo = np.linspace(min(x),max(x),100)
		y_hypo = results.predict(pd.DataFrame(data={dependant:x_hypo,independant:x_hypo}))
		ax.plot(x_hypo,y_hypo,color="r", label="hypothesis")

		# plot test data
		ax.scatter(x_test,y_test,marker = "x",color="k",label="test data")
	count += 1
plt.legend(loc="upper left")

# overwrite array with DataFrame
column_name = independant+"_vs_"+dependant
parameters = pd.DataFrame(data={"video_id":dataframe["video_id"],column_name:parameters})
print(parameters.head())
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
fig1, ax1 = plt.subplots()

test_sizes = [0.2,0.3,0.4,0.5]
for test_size in test_sizes:
	# select training data and randomize samples
	train_idx = np.random.rand(dataframe.no_of_entries.to_numpy()[0]) < test_size
	x = dataframe[independant[0]].to_numpy()[0]
	print(x)
	x = x[train_idx]
	y = dataframe[dependant].to_numpy()[0]
	y= y[train_idx]

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
	ax1.plot(x,results.fittedvalues,label=str(test_size))

	# show results
	print(results.summary())
plt.legend(loc="upper left")

plt.show()