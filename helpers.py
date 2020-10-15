import pandas as pd
import matplotlib.pyplot as plt

def time_series_from_dataset(df,column_label,sort_max=True):
	# make time series from trending videos
	trending_dates = df["trending_date"].drop_duplicates(keep="first").values

	# get data for time series
	time_series = pd.DataFrame()
	for date in trending_dates:
		trending_videos = df[df["trending_date"]==date]
		if sort_max:
			time_series = time_series.append(df.iloc[trending_videos[column_label].idxmax()])
		else:
			time_series = time_series.append(df.iloc[trending_videos[column_label].idxmin()])
	print(len(time_series.index))
	return time_series.set_index("trending_date")

def count_comments_ratings_errors_disabled(df):
	# make time series from trending videos
	trending_dates = df["trending_date"].drop_duplicates(keep="first").values

	# get data for time series
	time_series = pd.DataFrame(columns=["trending_date","comments_disabled","ratings_disabled","video_error_or_removed"])
	for row,date in enumerate(trending_dates):
		trending_videos = df[df["trending_date"]==date]
		time_series.loc[row] = [date,trending_videos["comments_disabled"].sum(),trending_videos["ratings_disabled"].sum(),trending_videos["video_error_or_removed"].sum()]
	return time_series.set_index("trending_date")


if __name__ == "__main__":

	WORKDIR = r"C:\Users\Marc\Documents\python_projects\machine_learning"
	dataframe = pd.read_csv(WORKDIR+r"\dataset\CAvideos.csv")
