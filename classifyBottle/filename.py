
import os
dir_path = '.\\res'
list =  [x for x in os.listdir(dir_path) if os.path.splitext(x)[1]=='.jpg']

for item in list:
	print(item)