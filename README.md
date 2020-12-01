# PotentialFinder - finding business potential in data lakes


# Abstract
Finding exponential growth trends and exponential growth potential is key to success in businesses and startups. Building a product for a market that can grow exponentially would increase the likelihood of success. These growth potentials can be found in a variety of sectors. Different challenges lie ahead in terms of finding exponential patterns and trends. This paper deals with finding these exponential patterns in data lakes. It also proposes different algorithms that can scale up to petabytes of data which can come in different sizes and formats (tabular files). These algorithms can be key to pattern discovery in data lakes, ultimately empowering our search for growth opportunities.


# Files Description
# Step0_data_processing
The scripts in this file are used to preprocess data.

# step0_kaggle_dataset_download
Downloads datasets from Kaggle datalake

# step1_generate_header
Generates headerfile.txt from the dataset folder which will be used by splitter.

# step2_sampling_localmachine
This code generates sample files in the local machine.

# Step2_samling_map_reduce
Generates sample file from the map reduce.

# satep3_splittling_partfiles_into_sample_files
Generates sample files from the mapreduce part files

# step3_potentialfinder
Does preprocessing for the dataset and does exponential and logistic pattern fit

# step5_potential_functions
It has support files for exponential and logistic fit and will be called by step3_potentialfinder

# step6_consolidation_for_ml_sampled_data
preprocessing for ML for the sampled data

# step6_consolidation_for_ml_whole_data
preprocessing for ML for the whole data


# step7_predicting_pattern_classes with ML
Has ML code for predicting classes

# step8_plots_classification
Generates graphs and plot

# Step9_creating_grphs
Generate graphs 




