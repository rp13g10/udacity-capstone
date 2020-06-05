'''Part 2 - Now the dataset is a bit cleaner and a churn flag is present, evaluate
how the dataset varies between accounts which churned/didn't churn.'''

####################################################################################################
# Script Initialization                                                                            #
####################################################################################################
# Import libraries
import os
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql import functions as ssf

# Import from module starting with a number
import importlib
p2 = importlib.import_module(
    '2_load_clean_data')

spark = p2.spark
script_env = p2.script_env
data_cleaned = p2.data_cleaned

if script_env == 'local':
    print("Navigate to 'localhost:4040' to check calculation progress.")

####################################################################################################
# Feature Engineering                                                                              #
####################################################################################################

r"""
One row per userId
Account age (max time since registration)
Gender
State
All following features aggregated over 2-3 different date ranges, to be determined
    No. of systems used
    Modal system
    404 error encountered
    No. distinct artists listened to
    No. songs listened to
    No. distinct songs listened to
    No. distinct sessions
    Mean items in session
    Max items in session
    Stdev items in session
    Mean time spent listening to each song
    Stdev time spent listening to each song
    Page - Add Friend
    Page - Thumbs up / No. songs
    Page - Thumbs down / No. songs
    Page - Help
    Page - Settings
    Level @ window start
    Level @ window end
    No. users with same surname & same location
    % of total plays accounted for by songs listened to by the  user
        how popular is the music they're listening to?
    Mean no. sessions per day
"""

# Determine sensible date ranges ###################################################################