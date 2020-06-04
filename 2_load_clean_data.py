'''Part 1 - Load and clean the provided dataset, deal with missing/invalid data'''
# pylint: disable=pointless-statement, pointless-string-statement

####################################################################################################
# Script Initialization                                                                            #
####################################################################################################
# Import libraries
import os
import pandas as pd
import pyspark.sql.functions as ssf
import pyspark.sql.types as sst
import re
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

if __name__ == '__main__':
    # Check available fields
    data_raw.printSchema()

    # Preview data
    data_raw.limit(5).toPandas()


# Health Check #####################################################################################

# Check common values & null counts ----------------------------------------------------------------

if __name__ == '__main__':
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


# Notes --------------------------------------------------------------------------------------------

r'''
Records with null userId might be useful, need to look at them in more detail
Some encoding errors, but that shouldn't impact the model
itemInSession looks useful for feature engineering
names will need removing
paid/free info from level will be useful
what is registration? looks like it might be a timestamp
http response codes in response could be useful
    307 is just a redirect, but 404 would be user-impacting
user agent could be used to get platform info
slightly more cancellations submitted than confirmed
significantly more downgrades started than confirmed
'''


# Null investigation -------------------------------------------------------------------------------

if __name__ == '__main__':
    # Records with missing user IDs
    data_null = data_raw.where(data_raw['userId'].isNull())
    null_sample = data_null.limit(1000).toPandas()
    null_sample
    # No results? Interesting...
    # Some userIds are populated with an empty string, not technically Null


    # Records with empty userIds
    data_empty = data_raw.where(data_raw['userId'] == '')
    empty_sample = data_empty.limit(1000).toPandas()
    empty_sample

    data_empty.groupby(['page', 'auth']).count().toPandas()
    # Looks like these are valid records, but without a userId they aren't useable


    # Records with missing song info
    # Probably a valid reason for this, but worth checking just in case
    data_null = data_raw.where(
        data_raw['userId'].isNotNull() & data_raw['song'].isNull())
    null_sample = data_null.limit(1000).toPandas()
    null_sample
    # These are still useful records, contains all data where the page isn't nextSong


# registration investigation -----------------------------------------------------------------------

if __name__ == '__main__':
    # Check theory that registration is just a timestamp
    data_reg = data_raw.where(data_raw['page'].isin({
        'Submit Registration',
        'Register'
    }))
    data_reg = data_reg.limit(1000).toPandas()
    data_reg

    # Pick a session ID which features in data_reg
    data_session = data_raw.where(data_raw['sessionId'] == 1719)
    data_session = data_session.limit(1000).toPandas()

    # Get actual registration time, compare to values in registration column
    reg_time = data_session.loc[data_session['page'] == 'Submit Registration', 'ts'].values[0]
    reg_min = data_session['registration'].min()
    reg_max = data_session['registration'].max()
    reg_time, reg_min, reg_max

    # View the time difference between actual & recorded registration time
    reg_time = pd.Timestamp(reg_time, unit='ms')
    reg_min = pd.Timestamp(reg_min, unit='ms')
    reg_max = pd.Timestamp(reg_max, unit='ms')
    reg_min - reg_time

    # Export the session info for evaluation
    if script_env == 'local':
        data_session.to_excel('summaries/sample_session.xlsx', index=False)

r'''Registration time is constant when a user is logged in, but it looks like there can
be a difference of several days between submitting a registration and the value
held in the registration column. Hypothesis is that ts could be sourced from the users
device while registration is a timestamp generated by the sparkify system. There
could also be a batch process updating the registration field?
Will tentatively try using the values in this column.

userAgent, userId, location, registration, gender, firstName, lastName can all be
filled in based on the sessionId, should improve data availability'''


####################################################################################################
# Data Cleaning                                                                                    #
####################################################################################################

# Fill in the gaps #################################################################################

# Get values for each session ID
data_gapfill = data_raw.groupby('sessionId').agg(
    # pylint: disable=no-member
    ssf.max('userAgent').alias('userAgent'),
    ssf.max('userId').alias('userId'),
    ssf.max('location').alias('location'),
    ssf.max('registration').alias('registration'),
    ssf.max('gender').alias('gender'),
    ssf.max('firstName').alias('firstName'),
    ssf.max('lastName').alias('lastName'),
)

# Drop these columns from the original dataset
data_cleaned = data_raw.drop(
    'userAgent', 'userId', 'location', 'registration', 'gender', 'firstName', 'lastName')

# Merge back to ensure fields are fully populated
data_cleaned = data_cleaned.join(data_gapfill, on='sessionId', how='inner')

# Check that no values are empty in session known to have null values
if __name__ == '__main__':
    data_cleaned.where(data_raw['sessionId'] == 1719).toPandas()


# Process text data ################################################################################

# Get platform from user agent string
platform_getter = ssf.regexp_extract(
    data_cleaned['userAgent'],
    r'[\w\/\.]+ \(([\w\s\.]+);.*\)',
    1
)

data_cleaned = data_cleaned.withColumn('platform', platform_getter)

# Get state code from location
def extract_state(location):
    # If multiple state codes given, just use the last one
    # Seems to happen with metropolitan areas which cross state lines?
    state = location.split(',')[-1]
    state = state.split('-')[-1].strip()
    return state

state_getter = ssf.udf(extract_state, sst.StringType())

data_cleaned = data_cleaned.withColumn('state', state_getter('location'))

# Remove extra tags from song titles, standardize case
# - e.g. Song Title [feat. artist] (Album Version) -->  SONG TITLE

def extract_song(song):

    if not isinstance(song, str):
        return None

    # Remove any trailing brackets
    song = re.sub(r' \[.+\]', '', song)
    song = re.sub(r' \(.+\)', '', song)

    # Remove any non-standard characters
    song = re.sub(r'[^\w\s]+', '', song)

    # Fix any duplicated spaces
    song = re.sub(r'\s\s+', ' ', song)

    # Standardize case, remove trailing whitespace
    song = song.strip().upper()
    return song

song_getter = ssf.udf(extract_song, sst.StringType())

data_cleaned = data_cleaned.withColumn('songCleaned', song_getter('song'))
