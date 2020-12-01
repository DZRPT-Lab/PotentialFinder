### This code is used to sample datasets on instance without using map reduce

### @authors: Abhijeet Amle, Roshan Bhandari, Abhimanyu Abhinav


from time import time

start_time = time()

line_count_file = '/mnt/disks/localdisk/data/linecount_40G_2.txt'
source_dir = "/mnt/disks/localdisk/data/data_40G_2/"
output_dir = "/mnt/disks/localdisk/sampled_output/40G_2/data_40G_2/"

file_processed = 0

with open(line_count_file, 'r') as file:
	for each_line in file.readlines():
		each_line = each_line.strip()
		
		count = each_line.split()[0]
		f_p_name = ' '.join(each_line.split()[1:])
		
		count = int(count)
		f_name = f_p_name.replace("data_40G_2/","")
		
		with open(output_dir + f_name, 'w') as file_sample, open(source_dir + f_name, 'r', encoding="iso-8859-1") as file_org:
			data = []
			for ctr, line in enumerate(file_org.readlines()):
				
				divisor = count // 5000
				
				if divisor == 0:
					divisor = 1
		
				if ctr % divisor == 0:
					data.append(line)

			file_sample.writelines(data)
		
		file_processed += 1
		if (file_processed % 500 == 0):
			print("Total files processed:", file_processed)
			
end_time = time()
t = (end_time - start_time) / 60
print("Total time in minutes:", t)

with open("/mnt/disks/localdisk/sampled_output/40G_2/" + "total_sampling_time_40G_2.txt", 'w') as file_time:
	file_time.writelines("Total time in minutes: ")
	file_time.writelines(str(t))
