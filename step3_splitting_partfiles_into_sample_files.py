#############
# @authors: Roshan Bhandari, Abhijeet Amle, Abhimanyu
# This code is used to split the part files botained from the 
# map reduce into multiple sample fi
#############

import os

data_dict = {}
header_data = {}

input_dir = "/mnt/disks/localdisk/new_sampled_data/"
output_file_dir = "/mnt/disks/localdisk/new_sampled_data1/"
header_file = "/mnt/disks/localdisk/headerfile.txt"

files = os.listdir(input_dir)
count = 0
with open(header_file) as file:
    for each_line in file.readlines():
        # print(each_line)
        try:
            filename = each_line.split(":")[0]
            header = ','.join(each_line.split(":")[1:])
            header_data[filename.replace("data/", "")] = header
        except Exception as e:
            # print(str(e))
           count += 1

    for each_file in files:
        with open (input_dir + each_file, 'r') as f:
            for each_line in f.readlines():
                #print(each_line)
                file, line = each_line.split("\t")
                file = file.strip()[1:-1]
                line = line[1:-2]
                # print(each_line.split())
                if file not in data_dict:
                    data_dict[file] = [line]
                else:
                    data_dict[file].append(line)
print("Total Unprocessed Files: ", count)
for each_file in data_dict:
    with open(output_file_dir + each_file, 'w') as file:
        file.write(header_data[each_file] + "\n")
        for count, each_line in enumerate(data_dict[each_file]):
            if header_data[each_file] != each_line:
                file.write(each_line + "\n")
