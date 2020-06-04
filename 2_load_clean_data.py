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
# Data Exploration                                                                                 #
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


# Schema exploration ###############################################################################

# Check available fields
data_raw.printSchema()

# Preview data
data_raw.limit(5).toPandas()


# Health Check #####################################################################################

# Check common values & null counts ----------------------------------------------------------------

total_rows = data_raw.count()
summaries = []
null_counts = []
for col in data_raw.columns:

    # Get no. of values 
    value_counts = data_raw.groupby(col).count()
    value_counts = value_counts.orderBy('count', ascending=False)

    # Make sure null count always comes through
    null_count = value_counts.where(value_counts[col].isNull())
    value_counts = value_counts.where(value_counts[col].isNotNull())

    # Convert output to Pandas df, row limit to prevent any memory issues
    value_counts = value_counts.limit(25).toPandas()
    null_count = null_count.toPandas()

    # Create summary dataframes for selected column
    summary = pd.concat([value_counts, null_count], axis=0, ignore_index=True)
    summary.columns = ['value', 'count']
    summary.loc[:, 'field'] = col
    summary = summary.sort_values(by='count', ascending=False)

    if null_count.empty:
        null_count = pd.DataFrame({
            'value': [None],
            'count': [0],
            'field': [col]
        })
    else:
        null_count.loc[:, 'field'] = col

    # Save output to memory
    summaries.append(summary)
    null_counts.append(null_count)

# Combine summary dataframes for each column
value_summary = pd.concat(summaries, axis=0, ignore_index=True)
null_summary = pd.concat(null_counts, axis=0, ignore_index=True)

# Calculate counts as percentages
value_summary.loc[:, 'percentage'] = value_summary['count']/total_rows
null_summary.loc[:, 'percentage'] = null_summary['count']/total_rows

# Standardize column order
value_summary = value_summary[['field', 'value', 'count', 'percentage']]
null_summary = null_summary[['field', 'value', 'count', 'percentage']]

# Write to disk if script is running locally
if script_env == 'local':
    value_summary.to_excel('summaries/value_summary.xlsx', index=False)
    null_summary.to_excel('summaries/null_summary.xlsx', index=False)
