import pandas as pd
import numpy as np
from datetime import datetime
import sys
import bs4 as bs
import requests

IMPORTFOLDER = r"C:\Users\Marc\Documents\python_projects\machine_learning"

df = pd.read_csv(IMPORTFOLDER+r"\dataset\DEvideos.csv")

# print some
print(len(df.index))
#print(datetime.strptime(df["trending_date"][0],"%y.%d.%m"))

print(df.columns)
print(df["video_id"][0])
print(df["tags"][0])

# input arguments for import
nargin = len(sys.argv)-1
argin = sys.argv[1:]

if nargin > 0 and argin[0] is "pickle":
	pass

# make time series from trending videos
trending_dates = df["trending_date"].drop_duplicates(keep="first").values

# get data for time series
time_series = pd.DataFrame()
for date in trending_dates:
	trending_videos = df[df["trending_date"]==date]
	time_series = time_series.append(df.iloc[trending_videos["comment_count"].idxmax()])
print(time_series["trending_date"])

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

