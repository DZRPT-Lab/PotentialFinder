#############
# @authors: Roshan Bhandari, Abhijeet Amle, Abhimanyu
# This code is used to sample datasets. It uses  Map Reduce framework for sampling dataset and runs on cluster of 
# machines including google dataproc.
# To Run call: python3 step1.py gs://bucket_potentialfinder/1-jan-30-june-2013-calls-for-service.csv --master-instance-type e2-highmem-8 --instance-type e2-highmem-4 --num-core-instance 4 --files ../linecount.txt --output-dir=gs://output_potential/out101.txt --core-instance-config '{"disk_config": {"boot_disk_size_gb": 100}}' -r dataproc
#############

import os
from mrjob.job import MRJob
from mrjob.step import MRStep

line_count_file = 'linecount.txt'

class MR5000DataSampler(MRJob):
    line_count = {}
    line_number = 0

    def mapper_init(self):
        with open(line_count_file, 'r') as file:
            for each_line in file.readlines():

                count = each_line.split()[0]
                f_p_name = ' '.join(each_line.split()[1:])

                count = int(count)
                f_name = f_p_name.replace("data/","")

               # count = int(each_line.replace("data/","").split()[0])
               # f_name = each_line.replace("data/","").split()[1]
                self.line_count[f_name] = count
        #pass

    def mapper(self, _, line):
        self.line_number += 1
        input_file = os.environ['mapreduce_map_input_file']
        f_name = input_file.split("/")
        input_file = f_name[len(f_name)-1]

        try:
            divisor = int(self.line_count[input_file] / 5000)
        except:
            divisor = 1

        if divisor == 0:
            divisor = 1

        if self.line_number % divisor == 0:
           yield (input_file, line)

    def reducer(self, key, values):
        for line in values:
            yield (key, line)

if __name__ == '__main__':
    MR5000DataSampler.run()
