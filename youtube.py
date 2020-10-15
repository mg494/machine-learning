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
WORKDIR = r"C:\Users\Marc\Documents\python_projects\machine_learning"

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

