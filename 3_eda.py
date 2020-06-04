'''Part 1 - Load and clean the provided dataset, deal with missing/invalid data'''

####################################################################################################
# Script Initialization                                                                            #
####################################################################################################
# Import libraries
import os
import pandas as pd
from pyspark.sql import SparkSession

# Set up a spark session
spark = SparkSession.builder.appName(
    'Sparkify'
).getOrCreate()


# User Inputs ######################################################################################

# One of: {'local', 'aws'}
script_env = 'local'

# Boolean
use_full_dataset = False


# Validate inputs ##################################################################################
assert script_env in {'local', 'aws'}, 'Invalid input for script_env'
assert isinstance(use_full_dataset, bool), 'Invalid input for use_full_dataset'



####################################################################################################
# Data Tidying                                                                                     #
####################################################################################################

# Load data ########################################################################################

if script_env == 'local':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    data_dir = 'data'
elif script_env == 'aws':
    data_dir = 's3n://udacity-dsnd/sparkify'

if use_full_dataset:
    data_file = 'sparkify_event_data.json'
else:
    data_file = 'mini_sparkify_event_data.json'

data_path = f"{data_dir}/{data_file}"

data_raw = spark.read.json(data_path)


# Fill in the gaps #################################################################################

