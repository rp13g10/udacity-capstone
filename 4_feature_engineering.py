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
from pyspark.sql import window as ssw
import pyspark

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

# Create target variable ###########################################################################

# Get churned customers, without bringing user IDs out of spark
# churned = data_cleaned.where(data_cleaned['churnFlag'] == 1).select('userId').distinct()
# churned = churned.withColumn(
#     #pylint: disable=no-member
#     'userChurnFlag', ssf.lit(1))
# data_cleaned = data_cleaned.join(churned, on='userId', how='left')
# data_cleaned = data_cleaned.fillna(0, subset=['userChurnFlag'])
# data_cleaned = data_cleaned.persist()

# # Trigger evaluation of the datset up to this point
# sample = data_cleaned.limit(1000).toPandas()
data_cleaned = data_cleaned.orderBy(
    'userID', 'ts', ascending=True
).persist()


# Static variables (no time dependency) ############################################################

static_vars = data_cleaned.groupby('userId').agg(
    #pylint: disable=no-member
    ssf.max('gender').alias('gender'),
    ssf.max('ts').alias('lastTs'),
    ssf.min('registration').alias('registration'),
    ssf.max('state').alias('state'),
    ssf.max('churnFlag').alias('userChurnFlag')
)

static_vars = static_vars.withColumn(
    'accountAge',
    static_vars['lastTs']-static_vars['registration'])

static_vars = static_vars.drop('lastTs', 'registration')

static_sample = static_vars.limit(1000).toPandas()

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

# Dynamic variables (evaluated over a time window) #################################################

# Get the max timestamp
limits = data_cleaned.agg(
    ssf.max('ts').alias('maxTs'),
    ssf.min('ts').alias('minTs')
).collect()[0]
max_ts = limits['maxTs']
min_ts = limits['minTs']

day_delta = 24 * 60 * 60 * 1000
week_delta = 7 * day_delta
month_delta = 31 * day_delta

start_times = {
    'week': max_ts - week_delta,
    'month': max_ts - month_delta
}

# outputs = []
# for name, start_ts in start_times.items():
    # Get subset of data
    # Calculate aggregate statistics
    # Scale where appropriate


# Modal system -------------------------------------------------------------------------------------
sys_counts = data_cleaned.groupby('userId', 'platform').agg(
    ssf.countDistinct('sessionId').alias('noSessions'))

# No built-in mode function, need to manually rank using window functions
# Using row_number rather than rank to avoid ties
usr_window = ssw.Window.partitionBy(
    sys_counts['userId']
).orderBy(
    sys_counts['noSessions'].desc()
)
sys_counts = sys_counts.withColumn(
    #pylint: disable=no-member
    'platformRank', ssf.row_number().over(usr_window)
)

# Take first ranked platform for each 
sys_modal = sys_counts.where(
    sys_counts['platformRank'] == 1
).select(
    'userId', 'platform'
)

# Simple aggregates --------------------------------------------------------------------------------

flagged = data_cleaned.withColumn(
    'httpError',
    ssf.when(data_cleaned['status']==404, 1).otherwise(0)
)
dynamic_vars = flagged.groupby(
    'userId'
).agg(
    #pylint: disable=no-member
    ssf.count('platform').alias('noPlatforms'),
    ssf.count('userAgent').alias('noSystems'),
    ssf.sum('httpError').alias('httpErrors'),
    ssf.countDistinct('artist').alias('noArtists'),
    ssf.countDistinct('songCleaned').alias('noSongs'),
    # Count excludes nulls, so this should be equivalent to counting
    # number of nextSong pages
    ssf.count('songCleaned').alias('noPlays'),
    ssf.first('level').alias('levelStart'),
    ssf.last('level').alias('levelEnd')
)


# Session stats ------------------------------------------------------------------------------------

# Limit to song plays
session_vars = data_cleaned.filter(data_cleaned['page'] == 'NextSong')

# Get play time for each song
session_window = ssw.Window.partitionBy(
    session_vars['userId'], session_vars['sessionId']
).orderBy(
    'ts'
)
session_vars = session_vars.withColumn(
    'nextPlayStart',
    ssf.lead(session_vars['ts']).over(session_window)
)

session_vars = session_vars.withColumn(
    'playTime',
    session_vars['nextPlayStart'] - session_vars['ts']
)

# Get play time for each session
session_vars = session_vars.groupby(
    'userId', 'sessionId'
).agg(
    ssf.sum('playTime').alias('sessionLength')
)

# Get play time stats for each user
session_vars = session_vars.groupby(
    'userId'
).agg(
    ssf.mean('sessionLength').alias('lengthMean'),
    ssf.stddev('sessionLength').alias('lengthStd'),
    ssf.sum('sessionLength').alias('lengthSum')
)


# Popularity score ---------------------------------------------------------------------------------

# For each song, how much do they contribute to the total number of plays?
song_data = data_cleaned.filter(data_cleaned['page'] == 'NextSong')
song_totals = song_data.groupby('songCleaned').agg(
    ssf.count('ts').alias('songTotal')
)
overall_total = song_totals.agg(
    ssf.sum('songTotal').alias('overallTotal')
).collect()[0]['overallTotal']

song_totals = song_totals.withColumn(
    'overallPerc',
    song_totals['songTotal'] / overall_total
).select(
    'songCleaned', 'overallPerc'
)

# For each user, how popular are the songs they're playing?
popularity = song_data.select('userId', 'songCleaned')

# Make a clone of popularity, bug in pyspark causes errors if this isn't done
# https://stackoverflow.com/questions/45713290/how-to-resolve-the-analysisexception-resolved-attributes-in-spark
popularity = spark.createDataFrame(popularity.rdd, popularity.schema)

popularity = popularity.join(
    song_totals,
    popularity['songCleaned']==song_totals['songCleaned'],
    how='left')
popularity = popularity.groupby('userId').agg(
    ssf.sum('overallPerc').alias('popularityScore')
)


# Clicks through to each page ----------------------------------------------------------------------
page_clicks = data_cleaned.groupby('userId').pivot('page').count()

# Ensure consistent column headers regardless of the dataset
included_pages = [
    'userId', 'About', 'Add Friend', 'Add to Playlist', 'Cancel', 'Downgrade',
    'Error', 'Help', 'Home', 'Login', 'Logout', 'NextSong', 'Register',
    'Roll Advert', 'Save Settings', 'Settings', 'Submit Downgrade', 'Submit Registration',
    'Submit Upgrade', 'Thumbs Down', 'Thumbs Up', 'Upgrade'
]

missing_pages = [x for x in included_pages if x not in page_clicks.columns]

# Add a column for any missing pages
for missing_page in missing_pages:
    page_clicks = page_clicks.withColumn(
        missing_page,
        ssf.lit(0)
    )

# Only bring through what's in the list, fill in any missing values with 0
page_clicks = page_clicks.select(*included_pages)
page_clicks = page_clicks.fillna(0)


# Bring everything together ------------------------------------------------------------------------

# Final transformations ############################################################################

# Dimensionality Reduction #########################################################################

# Scaling ##########################################################################################

# Determine sensible date ranges ###################################################################