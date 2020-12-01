### This code is used for generating header file for all datasets
### It will read all the datasets and write only the file name and header row in one file

### @authors: Abhijeet Amle, Roshan Bhandari, Abhimanyu Abhinav



import os

input_dir = "/mnt/d/Projects/Data_Mining/GCP_Bucket/simulate_data/"

with open ('headerfile.txt', 'w') as headerfile:
	for each_file in os.listdir(input_dir):
		with open(input_dir + each_file, 'r', encoding="iso-8859-1") as file:
			for count, each_line in enumerate(file.readlines()):
				if count == 0:
					headerfile.write(each_file + "###:###" + each_line)
				break