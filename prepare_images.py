import os,cv2,glob
import numpy as np

SOURCE = r"C:/Users/Marc/Documents/python_projects/machine_learning/thumbnails/"

win_file_search_string = os.path.join(SOURCE,"*/Thumbs.db")
for file in glob.glob(win_file_search_string):
	os.remove(file)
	print("removed",file)

file_search_string = os.path.join(SOURCE,"*/*.jpg")
thumbnails = glob.glob(file_search_string)
number_of_images = len(thumbnails)
print("number of images in dataset",number_of_images)

count_removed = 0
removed_files = []
for image_file in thumbnails:
	image = cv2.imread(image_file)
	if np.median(image) == 204.0:
		os.remove(image_file)
		count_removed += 1
		removed_files.append(image_file)

print(removed_files)
print("deleted",count_removed,"images")