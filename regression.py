import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import math,sys
from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import mse
from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
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
	x_train, x_test,y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=42, shuffle=True)

	# define model, a constant will be defined automatically as intercept
	formula_string = "{0} ~ {1} + np.sqrt({1}) + np.log({1})".format(dependant,independant)
	model = ols(formula=formula_string,data = pd.DataFrame(data={dependant:y_train,independant:x_train}))  #

	# results
	results = model.fit()
	parameters.append(np.array(results.params))

	if count == plot_index:
		print(x,y)

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


# overwrite array with DataFrame
column_name = independant+"_vs_"+dependant
parameters = pd.DataFrame(data={"video_id":dataframe["video_id"],column_name:parameters})
print(parameters.head())
parameters.to_pickle("./data/regression_parameters.pkl")

plt.savefig("./data/figures/regression_"+independant+"_"+dependant+".png")

# record change of mse
test_sizes = [0.5,0.4,0.3,0.2,0.1]
train_samples = []
train_mse = []
test_mse = []

# record for single video sample
x,y = dataframe[independant].iloc[plot_index],dataframe[dependant].iloc[plot_index]

# loop over test sizes
for test_size in test_sizes:
	# test size
	x_train, x_test,y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=42, shuffle=True)
	train_samples.append(len(x_train))

	# define model, a constant will be defined automatically as intercept
	formula_string = "{0} ~ {1} + np.sqrt({1}) + np.log({1})".format(dependant,independant)
	model = ols(formula=formula_string,data = pd.DataFrame(data={dependant:y_train,independant:x_train}))  #

	# results
	results = model.fit()
	train_mse.append(mse(y_train,results.fittedvalues.to_numpy()))

	y_pred = results.predict(pd.DataFrame(data={dependant:y_test,independant:x_test}))
	test_mse.append(mse(y_pred,y_test))

fig1,ax1 = plt.subplots()
ax1.plot(train_samples,train_mse,label="training")
ax1.plot(train_samples,test_mse,label = "test")
ax1.set_xlim(left=min(train_samples),right=max(train_samples))
plt.xticks(train_samples)
plt.xlabel("number of samples")
plt.ylabel("mean squared error")
plt.legend(loc="upper right")

"""
------- multi dimensional model ------------
"""

independant = ["views","likes"]
dependant = "comments"

formula_string = dependant +"~"
for idx,var in enumerate(independant):
	print(idx,var)
	if idx > 0:
		formula_string += "+ {0} + np.sqrt({0}) + np.log({0})".format(var)
	else:
		formula_string += "{0} + np.sqrt({0}) + np.log({0})".format(var)

# record for single video sample
x = dataframe[independant].iloc[plot_index]
y = dataframe[dependant].iloc[plot_index]
data = pd.DataFrame(data={dependant:y,independant[0]:x[0],independant[1]:x[1]})

#x_train, x_test,y_train, y_test
data_train,data_test = train_test_split(data, test_size=0.1, random_state=42, shuffle=True)
print(data_train)
model = ols(formula=formula_string,data = data_train)  #
results = model.fit()
results.summary()

# plot 3d
fig3 = plt.figure()
ax3 = fig3.gca(projection='3d')
#ax3 = Axes3D(fig)

sort_idx = np.argsort(results.fittedvalues)
z_fit = results.fittedvalues.to_numpy()[sort_idx]
x_fit = data_train.views.to_numpy()[sort_idx]
y_fit = data_train.likes.to_numpy()[sort_idx]

ax3.plot(x[0],x[1],y,label="training data")
ax3.plot(x_fit,y_fit,z_fit)

ax3.set_xlabel("views")
ax3.set_ylabel("likes")
ax3.set_zlabel("comments")



plt.show()
