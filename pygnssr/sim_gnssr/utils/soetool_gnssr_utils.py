import json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np


def loadJSONScheduleToPandasDataFrame(fileName):
    ''' Loads a JSON schedule to a Pandas DataFrame (GNSS-R events only)
    :param str fileName: File name of the schedule file to load
    :returns pandas.DataFrame: The loaded dataframe
    '''
    # Load the JSON schedule flattened to a Pandas dataframe
    jsonFile = json.load(open(fileName))
    df = json_normalize(jsonFile, [['command','rayGeolocation']],
        ['time',
        ['command', 'duration'],
        ['command', 'obsType'],
        ['command', 'rate'],
        ['command', 'system'],
        ['command', 'txId'],
        ['command', 'type']])

    # Convert the time columns to pandas format
    df['time'] = pd.to_datetime(df['time'])
    df['loctime'] = pd.to_datetime(df['loctime'])
    df['command.duration'] = pd.to_timedelta(df['command.duration'], unit='s')

    # Generate a unique track number (as they are not numbered in the JSON file)
    # This may not be the fastest approach but it ensures unique track numbers
    df['trackHash'] = df[['time', 'command.txId']].apply(lambda x: hash(tuple(x)),axis=1)
    df['trackNumber'] = (df['trackHash'].diff() != 0).cumsum() - 1

    # Include only GNSS-R events
    df = df[df['command.obsType'] == "gnssr"]
    df.reset_index()

    return df
