'''Part 2 - Now the dataset is a bit cleaner and a churn flag is present, evaluate
how the dataset varies between accounts which churned/didn't churn.'''

####################################################################################################
# Script Initialization                                                                            #
####################################################################################################
# Import libraries
import os
import pandas as pd
from pyspark.sql import SparkSession

# Import from module starting with a number
import importlib
p2 = importlib.import_module(
    '2_load_clean_data')

spark = p2.spark
script_env = p2.script_env
data_cleaned = p2.data_cleaned
data_cleaned.persist()



####################################################################################################
# EDA                                                                                              #
####################################################################################################

