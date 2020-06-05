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
# EDA                                                                                              #
####################################################################################################

# Create summary tables ############################################################################

# Get churned customers
churned = data_cleaned.where(data_cleaned['churnFlag'] == 1).select('userId').distinct()
churned = churned.toPandas()['userId'].tolist()
data_cleaned = data_cleaned.withColumn(
    'userChurnFlag',
    ssf.when(data_cleaned['userId'].isin(*churned), 1).otherwise(0))

data_cleaned = data_cleaned.persist()

# Simple Counts
user_stats = data_cleaned.groupby('userChurnFlag', 'userId').agg(
    #pylint: disable=no-member
    ssf.countDistinct('artist').alias('noArtists'),
    ssf.countDistinct('song').alias('noSongs'),
    ssf.count('song').alias('noPlays'),
    ssf.max('gender').alias('gender'),
    ssf.max('state').alias('state'),
    ssf.max('registration').alias('registration'),
    ssf.mean('length').alias('meanSongLength')
)
user_stats = user_stats.toPandas()
user_stats.loc[:, 'registration'] = user_stats['registration'].map(
    lambda x: pd.Timestamp(x, unit='ms'))

# Items in Session
item_stats = data_cleaned.groupby('userChurnFlag', 'sessionId').agg(
    #pylint: disable=no-member
    ssf.max('itemInSession').alias('sessionLength'),
    ssf.max('platform').alias('platform')
)
item_stats = item_stats.toPandas()

# Page visits
page_stats = data_cleaned.groupby('userChurnFlag', 'page').count()
page_stats = page_stats.toPandas()

# Status codes
status_stats = data_cleaned.groupby('userChurnFlag', 'status').count()
status_stats = status_stats.toPandas()


# Generate plots ###################################################################################

# Simple counts (continuous)
id_cols = {'userChurnFlag', 'userId'}
cat_cols = {'gender', 'state'}
for user_col in [x for x in user_stats.columns if x not in id_cols.union(cat_cols)]:
    fig = px.histogram(
        user_stats,
        x=user_col,
        color='userChurnFlag',
        barmode='overlay',
        histnorm='percent',
        labels={0: 'No Churn', 1: 'Churn'},
        nbins=50
        )
    fig.write_html(f'figures/{user_col}StatsPerc.html')

    fig = px.histogram(
        user_stats,
        x=user_col,
        color='userChurnFlag',
        barmode='overlay',
        labels={0: 'No Churn', 1: 'Churn'},
        nbins=50)
    fig.write_html(f'figures/{user_col}StatsAbs.html')

# Simple counts (discrete)
group_counts = user_stats.groupby('userChurnFlag').agg(
    totalUsers = ('userId', 'nunique')
).reset_index()
for user_col in cat_cols:
    plot_df = user_stats.groupby(['userChurnFlag', user_col]).agg(
        noUsers = ('userId', 'nunique')
    ).reset_index()
    plot_df = plot_df.merge(group_counts, on='userChurnFlag', how='inner')
    plot_df.loc[:, 'percUsers'] = plot_df['noUsers'] / plot_df['totalUsers']
    plot_df = plot_df.sort_values(by='noUsers', ascending=False)
    plot_df.loc[:, 'userChurnFlag'] = plot_df['userChurnFlag'].astype(bool)

    fig = px.bar(
        plot_df,
        x=user_col,
        y='percUsers',
        color='userChurnFlag',
        barmode='group',
        labels={True: 'Churn', False: 'No Churn'}
    )
    fig.write_html(f'figures/{user_col}StatsPerc.html')

    fig = px.bar(
        plot_df,
        x=user_col,
        y='noUsers',
        color='userChurnFlag',
        barmode='group',
        labels={True: 'Churn', False: 'No Churn'}
    )
    fig.write_html(f'figures/{user_col}StatsAbs.html')

# Session length
fig = px.histogram(
    item_stats,
    x='sessionLength',
    color='userChurnFlag',
    barmode='overlay',
    histnorm='percent',
    labels={0: 'No Churn', 1: 'Churn'},
    nbins=50)
fig.write_html('figures/sessionLengthStatsPerc.html')

fig = px.histogram(
    item_stats,
    x='sessionLength',
    color='userChurnFlag',
    barmode='overlay',
    labels={0: 'No Churn', 1: 'Churn'},
    nbins=50)
fig.write_html('figures/sessionLengthStatsAbs.html')

# Platform
fig = px.histogram(
    item_stats,
    x='platform',
    color='userChurnFlag',
    barmode='group',
    histnorm='percent',
    labels={0: 'No Churn', 1: 'Churn'}
)
fig.write_html('figures/platformStatsPerc.html')

fig = px.histogram(
    item_stats,
    x='platform',
    color='userChurnFlag',
    barmode='group',
    labels={0: 'No Churn', 1: 'Churn'}
)
fig.write_html('figures/platformStatsAbs.html')


# Page visits
page_stats.loc[:, 'userChurnFlag'] = page_stats['userChurnFlag'].astype(bool)
group_sums = page_stats.groupby('userChurnFlag').agg(
    totalVisits = ('count', 'sum')
).reset_index()
page_stats = page_stats.merge(group_sums, on='userChurnFlag', how='inner')
page_stats.loc[:, 'percentage'] = page_stats['count'] / page_stats['totalVisits']
page_stats = page_stats.sort_values(by='count', ascending=False)

fig = px.bar(
    page_stats,
    x='page',
    y='count',
    color='userChurnFlag',
    barmode='group',
)
fig.write_html('figures/pageStatsAbs.html')

fig = px.bar(
    page_stats,
    x='page',
    y='percentage',
    color='userChurnFlag',
    barmode='group',
)
fig.write_html('figures/pageStatsPerc.html')

# Status codes
status_stats.loc[:, 'userChurnFlag'] = status_stats['userChurnFlag'].astype(bool)
group_sums = status_stats.groupby('userChurnFlag').agg(
    totalResponses = ('count', 'sum')
).reset_index()
status_stats = status_stats.merge(group_sums, on='userChurnFlag', how='inner')
status_stats.loc[:, 'percentage'] = status_stats['count'] / status_stats['totalResponses']
status_stats = status_stats.sort_values(by='count', ascending=False)

fig = px.bar(
    status_stats,
    x='status',
    y='count',
    color='userChurnFlag',
    barmode='group',
)
fig.write_html('figures/statusStatsAbs.html')

fig = px.bar(
    status_stats,
    x='status',
    y='percentage',
    color='userChurnFlag',
    barmode='group',
)
fig.write_html('figures/statusStatsPerc.html')
