import pandas as pd
import numpy as np
from datetime import datetime
import bs4 as bs
import requests, os, sys, glob

# local packages
from helpers import *

# input arguments for import
nargin = len(sys.argv)-1
argin = sys.argv[1:]

# parameter
SOURCE = r"C:\Users\Marc\Documents\python_projects\machine_learning"

# merge country data from source to numerical dataset
if nargin > 0 and argin[0] == "merge":
	all_countries_numerical = pd.DataFrame()
	files = glob.glob(SOURCE+r"\dataset\*.csv")
	for file in files:
		filepath, filename = os.path.split(file)
		country_data = pd.read_csv(file)
		country_data.drop(["title","channel_title","tags","thumbnail_link","description"],axis="columns",inplace = True)
		country_data["country"] = filename[0:2]
		all_countries_numerical = all_countries_numerical.append(country_data)

	time_to_trends = []
	for publish_string, trending_string in zip(all_countries_numerical["publish_time"].values,all_countries_numerical["trending_date"].values):
		publish_date = datetime.strptime(publish_string,"%Y-%m-%dT%H:%M:%S.%fZ").date()
		trending_date = datetime.strptime(trending_string,"%y.%d.%m").date()
		delta = trending_date - publish_date
		time_to_trends.append(delta.days)
	all_countries_numerical["days_to_trends"] = time_to_trends
	all_countries_numerical.reset_index()
	all_countries_numerical.to_pickle("./dataset.pkl")


# write time series from dataset
if nargin > 0 and argin[0] == "timeseries":
	time_series = pd.DataFrame()
	files = glob.glob(WORKDIR+r"\dataset\*.csv")
	for file in files:
		print(file)
		dataframe = pd.read_csv(file)
		print(len(dataframe.index))
		print(dataframe.columns)
		time_series = time_series_from_dataset(dataframe,"likes")




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

