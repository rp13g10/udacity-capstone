'''Quick script to fetch sample data from s3 for development purposes.
Initial development to be carried out on a local pyspark cluster to
minimize AWS charges.'''

import boto3

# Credentials need to be set via environment variables
s3 = boto3.client('s3')

# Mini dataset to work with
with open('data/mini_sparkify_event_data.json', 'wb') as target:
    s3.download_fileobj(
        'udacity-dsnd',
        'sparkify/mini_sparkify_event_data.json',
        target)

# Just for a laugh...
with open('data/sparkify_event_data.json', 'wb') as target:
    s3.download_fileobj(
        'udacity-dsnd',
        'sparkify/sparkify_event_data.json',
        target)