import pandas as pd
from datetime import datetime
import os, sys, glob
import matplotlib.pyplot as plt
from progress.bar import Bar

# package settings
pd.set_option('display.max_columns', None)

# input arguments for import
nargin = len(sys.argv)-1
argin = sys.argv[1:]

# parameter
SOURCE = r"C:\Users\Marc\Documents\python_projects\machine_learning"

# make numerical dataset from raw data
if nargin > 0 and argin[0] == "dataset":
	# merge all countries in one DataFrame and drop non numerical data
	all_countries_numerical = pd.DataFrame()
	files = glob.glob(SOURCE+r"\dataset\*.csv")
	for file in files:
		filepath, filename = os.path.split(file)
		country_data = pd.read_csv(file)
		country_data.drop(["title","channel_title","tags","thumbnail_link","description"],axis="columns",inplace = True)
		country_data["country"] = filename[0:2]
		all_countries_numerical = all_countries_numerical.append(country_data)

	print(len(all_countries_numerical.index))

	# fix trending_date format and extract date from publish time
	all_countries_numerical.reset_index(inplace=True)
	all_countries_numerical["trending_date"] = pd.to_datetime(all_countries_numerical["trending_date"],format="%y.%d.%m")
	publish_time = pd.to_datetime(all_countries_numerical["publish_time"]).dt.time
	publish_date = pd.to_datetime(all_countries_numerical["publish_time"]).dt.date
	all_countries_numerical["publish_time"] = publish_time
	all_countries_numerical["publish_date"] = publish_date

	# calculate time_to_trends for each unique video
	unique_videos = all_countries_numerical.drop_duplicates("video_id")
	unique_trending_dates = unique_videos["trending_date"].values.astype("datetime64[D]")
	unique_publish_dates = unique_videos["publish_date"].values.astype("datetime64[D]")
	time_to_trends = (unique_trending_dates - unique_publish_dates).tolist()
	video_list = unique_videos["video_id"].to_numpy().tolist()

	# put time_to_trends in all_countries_numerical
	time_to_trends_all = [None] * len(all_countries_numerical.index)
	time_to_trends_all = [time_to_trends[video_list.index(video_id)] for video_id in all_countries_numerical["video_id"]]

	all_countries_numerical["time_to_trends"] = time_to_trends_all
	all_countries_numerical.to_pickle("./dataset.pkl")

if nargin > 0 and argin[0] == "timeseries":
	# make timeseries from dataframe
	time_series = pd.DataFrame()

	# read dataset
	dataframe = pd.read_pickle("./dataset.pkl")

	for country in dataframe["country"].unique():
		# filter frame by country
		df_by_country = dataframe[dataframe["country"]==country]

		# get dates to iterate over
		trending_dates = df_by_country["trending_date"].drop_duplicates(keep="first").values

		# iterate over each day and do the things with numerical data
		mean_views = []
		mean_likes = []
		mean_dislikes = []
		mean_comments = []
		sum_disabled_comments = []
		sum_disabled_ratings = []
		sum_deleted_videos = []
		for date in trending_dates:
			trending_videos = df_by_country[df_by_country["trending_date"]==date]

			# calculate mean views, comments, likes and dislikes
			mean_views.append(trending_videos["views"].mean())
			mean_comments.append(trending_videos["comment_count"].mean())
			mean_likes.append(trending_videos["likes"].mean())
			mean_dislikes.append(trending_videos["dislikes"].mean())

			# calculate sum of disabled comments or rating and deleted videos
			sum_disabled_comments.append(trending_videos["comments_disabled"].sum())
			sum_disabled_ratings.append(trending_videos["ratings_disabled"].sum())
			sum_deleted_videos.append(trending_videos["video_error_or_removed"].sum())

		time_series = time_series.append(pd.DataFrame(data={ "trending_date": trending_dates,
																		"country": [country] *len(trending_dates),
																		"mean_likes":mean_likes,
																		"mean_views":mean_views,
																		"mean_comments":mean_comments,
																		"sum_disabled_comments":sum_disabled_comments,
																		"sum_disabled_ratings":sum_disabled_ratings,
																		"sum_deleted_videos":sum_deleted_videos}))

	time_series.sort_values("trending_date",inplace=True)
	time_series.to_pickle("./timeseries.pkl")

# read dataset from pickle
dataframe = pd.read_pickle("./dataset.pkl")
print(len(dataframe.video_id),len(dataframe.video_id.unique()))

# read time series from pickle
timeseries = pd.read_pickle("./timeseries.pkl")

# plot mean views for each country
plotitem = "mean_likes"
fig, axis = plt.subplots()
for country in timeseries["country"].unique():
	df_filtered = timeseries[timeseries["country"]==country]
	axis.plot(df_filtered["trending_date"].values,df_filtered[plotitem].values, label=country)

plt.legend(loc="best")
plt.title(plotitem)
plt.savefig("timeseries_"+plotitem+".png")


"""
# get html text from video
page = requests.get("http://www.youtube.com/watch?v="+df["video_id"][0])

# convert to text
soupObj = bs.BeautifulSoup(page.text, "html.parser")

# finding meta info for title
title = soupObj.find("span", class_="watch-title") #.text.replace("\n", "")
print(title)
# finding meta info for views
views = soupObj.find("div", class_="watch-view-count").text

# finding meta info for likes
likes = soupObj.find("span", class_="like-button-renderer").span.button.text

print(text)
"""

