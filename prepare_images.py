import os,cv2
import numpy as np

SOURCE = r"C:/Users/Marc/Documents/python_projects/machine_learning/thumbnails/25/"

thumbnails = os.listdir(SOURCE)
number_of_images = len(thumbnails)
try:
	thumbnails.remove("Thumbs.db")
except ValueError:
	pass

count_removed = 0
removed_files = []
for image_file in thumbnails:
	print(image_file)
	image = cv2.imread(SOURCE+r'/'+image_file)
	if np.median(image) == 204.0:
		os.remove(SOURCE+r'/'+image_file)
		count_removed += 1
		removed_files.append(image_file)
print("deleted",count_removed,"images")
print(removed_files)