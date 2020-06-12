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
from pyspark.sql import types as sst
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

# # Trigger evaluation of the datset up to this point
data_cleaned = data_cleaned.orderBy(
    'userID', 'ts', ascending=True
).persist()
data_sample = data_cleaned.limit(1000).toPandas()

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

# Dynamic variables (evaluated over a time window) #################################################

# Modal system -------------------------------------------------------------------------------------

def get_modal_system(data_in, name):
    sys_counts = data_in.groupby('userId', 'platform').agg(
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
        'userId', f'platform'
    )

    sys_modal = sys_modal.withColumnRenamed('platform', f'platform{name}')

    return sys_modal

# Simple aggregates --------------------------------------------------------------------------------

def get_simple_aggregates(data_in, name):

    flagged = data_in.withColumn(
        'httpError',
        ssf.when(data_in['status']==404, 1).otherwise(0)
    )
    dynamic_vars = flagged.groupby(
        'userId'
    ).agg(
        #pylint: disable=no-member
        ssf.count('platform').alias(f'noPlatforms{name}'),
        ssf.count('userAgent').alias(f'noSystems{name}'),
        ssf.sum('httpError').alias(f'httpErrors{name}'),
        ssf.countDistinct('artist').alias(f'noArtists{name}'),
        ssf.countDistinct('songCleaned').alias(f'noSongs{name}'),
        # Count excludes nulls, so this should be equivalent to counting
        # number of nextSong pages
        ssf.count('songCleaned').alias(f'noPlays{name}'),
        ssf.first('level').alias(f'levelStart{name}'),
        ssf.last('level').alias(f'levelEnd{name}')
    )

    return dynamic_vars


# Session stats ------------------------------------------------------------------------------------

def get_session_stats(data_in, name):
    # Limit to song plays
    session_vars = data_in.filter(data_in['page'] == 'NextSong')

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
        #pylint: disable=no-member
        ssf.sum('playTime').alias('sessionLength')
    )

    # Get play time stats for each user
    session_vars = session_vars.groupby(
        'userId'
    ).agg(
        #pylint: disable=no-member
        ssf.mean('sessionLength').alias(f'lengthMean{name}'),
        ssf.stddev('sessionLength').alias(f'lengthStd{name}'),
        ssf.sum('sessionLength').alias(f'lengthSum{name}')
    )

    return session_vars


# Popularity score ---------------------------------------------------------------------------------

def get_popularity_scores(data_in, name):
    # For each song, how much do they contribute to the total number of plays?
    song_data = data_in.filter(data_in['page'] == 'NextSong')
    song_totals = song_data.groupby('songCleaned').agg(
        #pylint: disable=no-member
        ssf.count('ts').alias('songTotal')
    )
    overall_total = song_totals.agg(
        #pylint: disable=no-member
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
        #pylint: disable=no-member
        ssf.sum('overallPerc').alias(f'popularityScore{name}')
    )

    return popularity


# Clicks through to each page ----------------------------------------------------------------------

def get_page_clicks(data_in, name):

    def format_page(page):
        if not isinstance(page, str):
            return None
        
        page = page.replace(' ', '')
        page = f'{page}{name}'
        return page
    
    page_getter = ssf.udf(format_page, sst.StringType())

    page_clicks = data_in.withColumn('page', page_getter('page'))

    page_clicks = page_clicks.groupby('userId').pivot('page').count()

    # Ensure consistent column headers regardless of the dataset
    included_pages = [
        'About', 'Add Friend', 'Add to Playlist', 'Cancel', 'Downgrade',
        'Error', 'Help', 'Home', 'Login', 'Logout', 'NextSong', 'Register',
        'Roll Advert', 'Save Settings', 'Settings', 'Submit Downgrade', 'Submit Registration',
        'Submit Upgrade', 'Thumbs Down', 'Thumbs Up', 'Upgrade'
    ]

    included_pages = ['userId'] + [format_page(x) for x in included_pages]

    missing_pages = [x for x in included_pages if x not in page_clicks.columns]

    # Add a column for any missing pages
    for missing_page in missing_pages:
        page_clicks = page_clicks.withColumn(
            #pylint: disable=no-member
            missing_page,
            ssf.lit(0)
        )

    # Only bring through what's in the list
    page_clicks = page_clicks.select(*included_pages)

    return page_clicks


# Generate all required tables ---------------------------------------------------------------------

# Get the max timestamp
limits = data_cleaned.agg(
    #pylint: disable=no-member
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

outputs = None
for name, start_ts in start_times.items():

    name = name.title()

    # Get subset of data
    data_subset = data_cleaned.where(
        data_cleaned['ts'] >= start_ts
    ).persist()

    modal_system = get_modal_system(data_subset, name)

    simple_aggregates = get_simple_aggregates(data_subset, name)

    session_stats = get_session_stats(data_subset, name)

    popularity_scores = get_popularity_scores(data_subset, name)

    page_clicks = get_page_clicks(data_subset, name)

    merged = modal_system.join(
        simple_aggregates,
        on='userId',
        how='full_outer'
    ).join(
        session_stats,
        on='userId',
        how='full_outer'
    ).join(
        popularity_scores,
        on='userId',
        how='full_outer'
    ).join(
        page_clicks,
        on='userId',
        how='full_outer'
    ).persist()

    if outputs is None:
        outputs = merged
    else:
        outputs = outputs.join(merged, on='userId', how='full_outer')

merged = static_vars.join(
    outputs,
    on='userId',
    how='full_outer'
)

platform_cols = [x for x in outputs.columns if 'platform' in x]
level_cols = [x for x in outputs.columns if 'level' in x]
merged = merged.fillna('unknown', subset=platform_cols)
merged = merged.fillna('free', subset=level_cols)
merged = merged.fillna(0)

encoded = merged.persist()
merged = merged.persist()

merged_sample = merged.limit(1000).toPandas()

# Final transformations ############################################################################
from pyspark.ml import feature as smf

# One-hot encoding of categorical variables --------------------------------------------------------

cat_cols = ['gender', 'state', 'platform', 'level']
cat_cols = [x for x in encoded.columns if any((y for y in cat_cols if y in x))]

def fill_empty_string(string_in, fill_value='unknown'):
    if not isinstance(string_in, str):
        return fill_value
    elif not string_in:
        return fill_value
    else:
        return string_in

na_handler = ssf.udf(fill_empty_string, sst.StringType())

indexers = {}
for cat_col in cat_cols:
    encoded = encoded.withColumn(cat_col, na_handler(cat_col))
    indexer = smf.StringIndexer(
        inputCol=cat_col,
        outputCol=f"{cat_col}Inx"
    )
    indexer = indexer.fit(encoded)
    encoded = indexer.transform(encoded)
    encoded = encoded.drop(cat_col).withColumnRenamed(f"{cat_col}Inx", cat_col)
    indexers[cat_col] = indexer


encoder = smf.OneHotEncoderEstimator(
    inputCols=cat_cols,
    outputCols=cat_cols
)
encoder = encoder.fit(encoded)
encoded = encoder.transform(encoded)
encoded.show()


# Dimensionality Reduction #########################################################################



# Scaling ##########################################################################################



# Determine sensible date ranges ###################################################################
