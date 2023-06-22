"""
Objective event classification using gust ratio

Classify wind gust events into convective or non-convective based on the wind
gust ratio, as defined in El Rafei et al. 2023

If r_1 = V_G/V_1 < 2.0 and r_2 = V_G/V_2 < 2.0, then the event is considered
synoptic. Otherwise, the event is classed as convective

V_G = peak gust wind speed
V_1 = mean gust wind speed for the 2 hour time period before the gust event
V_2 = mean gust wind speed for the 2 hour time period after the gust event.

El Rafei, M., S. Sherwood, J. Evans, and A. Dowdy, 2023: Analysis and
characterisation of extreme wind gust hazards in New South Wales,
Australia. *Nat Hazards*, **117**, 875–895,
https://doi.org/10.1007/s11069-023-05887-1.

"""

import os
import re
import glob
import argparse
import logging
from os.path import join as pjoin
from datetime import datetime, timedelta
from configparser import ConfigParser, ExtendedInterpolation
import pandas as pd
import numpy as np
import warnings

from process import pAlreadyProcessed, pWriteProcessedFile, pArchiveFile, pInit
from files import flStartLog, flGetStat, flSize
from stndata import ONEMINUTESTNNAMES, ONEMINUTEDTYPE, ONEMINUTENAMES

warnings.simplefilter('ignore', RuntimeWarning)
pd.set_option('mode.chained_assignment', None)

LOGGER = logging.getLogger()
PATTERN = re.compile(r".*Data_(\d{6}).*\.txt")
STNFILE = re.compile(r".*StnDet.*\.txt")
TZ = {"QLD": 10, "NSW": 10, "VIC": 10,
      "TAS": 10, "SA": 9.5, "NT": 9.5,
      "WA": 8, "ANT": 0}


def start():
    """
    Parse command line arguments, initiate processing module (for tracking
    processed files) and start the main loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help="Configuration file")
    parser.add_argument('-v', '--verbose', help="Verbose output",
                        action='store_true')
    args = parser.parse_args()

    configFile = args.config_file
    verbose = args.verbose
    config = ConfigParser(allow_no_value=True,
                          interpolation=ExtendedInterpolation())
    config.optionxform = str
    config.read(configFile)

    pInit(configFile)
    main(config, verbose)


def main(config, verbose=False):
    """
    Start logger and call the loop to process source files.

    :param config: `ConfigParser` object with configuration loaded
    :param boolean verbose: If `True`, print logging messages to STDOUT

    """

    logfile = config.get('Logging', 'LogFile')
    loglevel = config.get('Logging', 'LogLevel', fallback='INFO')
    verbose = config.getboolean('Logging', 'Verbose', fallback=verbose)
    datestamp = config.getboolean('Logging', 'Datestamp', fallback=False)
    LOGGER = flStartLog(logfile, loglevel, verbose, datestamp)

    ListAllFiles(config)
    processStationFiles(config)
    processFiles(config)


def ListAllFiles(config):
    """
    For each item in the 'Categories' section of the configuration file, load
    the specification (glob) for the files, then pass to `expandFileSpecs`

    :param config: `ConfigParser` object

    Example:

    [Categories]
    1=CategoryA
    2=CategoryB

    [CategoryA]
    OriginDir=C:/inputpath/
    Option2=OtherValue
    *.csv
    *.zip


    """
    global g_files
    g_files = {}
    categories = config.items('Categories')
    for idx, category in categories:
        specs = []
        items = config.items(category)
        for k, v in items:
            if v == '':
                specs.append(k)
        expandFileSpecs(config, specs, category)


def expandFileSpec(config, spec, category):
    """
    Given a file specification and a category, list all files that match the
    spec and add them to the :dict:`g_files` dict.
    The `category` variable corresponds to a section in the configuration file
    that includes an item called 'OriginDir'.
    The given `spec` is joined to the `category`'s 'OriginDir' and all matching
    files are stored in a list in :dict:`g_files` under the `category` key.

    :param config: `ConfigParser` object
    :param str spec: A file specification. e.g. '*.*' or 'IDW27*.txt'
    :param str category: A category that has a section in the source
    configuration file
    """
    if category not in g_files:
        g_files[category] = []

    origindir = config.get(category, 'OriginDir',
                           fallback=config.get('Defaults', 'OriginDir'))
    spec = pjoin(origindir, spec)
    files = glob.glob(spec)
    LOGGER.info(f"{len(files)} {spec} files to be processed")
    for file in files:
        if os.stat(file).st_size > 0:
            if file not in g_files[category]:
                g_files[category].append(file)


def expandFileSpecs(config, specs, category):
    for spec in specs:
        expandFileSpec(config, spec, category)


def processStationFiles(config):
    """
    Process all the station files to populate a global station file DataFrame

    :param config: `ConfigParser` object
    """

    global g_files
    global g_stations
    global LOGGER

    stnlist = []
    category = 'Stations'
    for f in g_files[category]:
        LOGGER.info(f"Processing {f}")
        stnlist.append(getStationList(f))
    g_stations = pd.concat(stnlist)


def getStationList(stnfile: str) -> pd.DataFrame:
    """
    Extract a list of stations from a station file

    :param str stnfile: Path to a station file

    :returns: :class:`pd.DataFrame`
    """
    LOGGER.debug(f"Retrieving list of stations from {stnfile}")
    df = pd.read_csv(stnfile, sep=',', index_col='stnNum',
                     names=ONEMINUTESTNNAMES,
                     keep_default_na=False,
                     converters={
                         'stnName': str.strip,
                         'stnState': str.strip
                         })
    LOGGER.debug(f"There are {len(df)} stations")
    return df


def processFiles(config):
    """
    Process a list of files in each category
    """
    global g_files
    global LOGGER
    unknownDir = config.get('Defaults', 'UnknownDir')
    defaultOriginDir = config.get('Defaults', 'OriginDir')
    deleteWhenProcessed = config.getboolean(
        'Files', 'DeleteWhenProcessed', fallback=False)
    archiveWhenProcessed = config.getboolean(
        'Files', 'ArchiveWhenProcessed', fallback=True)
    outputDir = config.get('Output', 'Path', fallback=unknownDir)
    LOGGER.debug(f"DeleteWhenProcessed: {deleteWhenProcessed}")
    LOGGER.debug(f"Output directory: {outputDir}")
    if not os.path.exists(unknownDir):
        os.mkdir(unknownDir)

    if not os.path.exists(pjoin(outputDir, 'gustratio')):
        os.makedirs(pjoin(outputDir, 'gustratio'))

    category = "Input"
    originDir = config.get(category, 'OriginDir',
                           fallback=defaultOriginDir)
    LOGGER.debug(f"Origin directory: {originDir}")

    for f in g_files[category]:
        LOGGER.info(f"Processing {f}")
        directory, fname, md5sum, moddate = flGetStat(f)
        if pAlreadyProcessed(directory, fname, "md5sum", md5sum):
            LOGGER.info(f"Already processed {f}")
        else:
            if processFile(f, outputDir):
                LOGGER.info(f"Successfully processed {f}")
                pWriteProcessedFile(f)
                if archiveWhenProcessed:
                    pArchiveFile(f)
                elif deleteWhenProcessed:
                    os.unlink(f)


def processFile(filename: str, outputDir: str) -> bool:
    """
    process a file and store output in the given output directory

    :param str filename: path to a station data file
    :param str outputDir: Output path for data & figures to be saved
    """

    global g_stations

    LOGGER.info(f"Loading data from {filename}")
    LOGGER.debug(f"Data will be written to {outputDir}")
    m = PATTERN.match(filename)
    stnNum = int(m.group(1))
    stnState = g_stations.loc[stnNum, 'stnState']
    stnName = g_stations.loc[stnNum, 'stnName']
    LOGGER.info(f"{stnName} - {stnNum} ({stnState})")
    filesize = flSize(filename)
    if filesize == 0:
        LOGGER.warning(f"Zero-sized file: {filename}")
        rc = False
    else:
        basename = f"{stnNum:06d}.pkl"
        dfmax = extractGustRatio(filename, stnState,)
        if dfmax is None:
            LOGGER.warning(f"No data returned for {filename}")
        else:
            outputFile = pjoin(outputDir, 'gustratio', basename)
            LOGGER.debug(f"Writing data to {outputFile}")
            dfmax.to_pickle(outputFile)

        rc = True
    return rc


def extractGustRatio(filename, stnState, variable='windgust'):
    """
    Extract daily maximum value of `variable` from 1-minute observation records
    contained in `filename` and evaluate gust ratio

    :param filename: str, path object or file-like object
    :param stnState: str, station State (for determining local time zone)
    :param str variable: the variable to extract daily maximum values
         default "windgust"

    :returns: `pandas.DataFrame`

    """

    LOGGER.debug(f"Reading station data from {filename}")
    try:
        df = pd.read_csv(filename, sep=',', index_col=False,
                         dtype=ONEMINUTEDTYPE,
                         names=ONEMINUTENAMES,
                         header=0,
                         parse_dates={'datetime': [7, 8, 9, 10, 11]},
                         na_values=['####'],
                         skipinitialspace=True)
    except Exception as err:
        LOGGER.exception(f"Cannot load data from {filename}: {err}")
        return None

    LOGGER.debug("Filtering on quality flags")
    for var in ['temp', 'temp1max', 'temp1min', 'wbtemp',
                'dewpoint', 'rh', 'windspd', 'windmin',
                'winddir', 'windsd', 'windgust', 'mslp', 'stnp']:
        df.loc[~df[f"{var}q"].isin(['Y']), [var]] = np.nan

    # Hacky way to convert from local standard time to UTC:
    df['datetime'] = pd.to_datetime(df.datetime, format="%Y %m %d %H %M")
    LOGGER.debug("Converting from local to UTC time")
    df['datetime'] = df.datetime - timedelta(hours=TZ[stnState])
    df['date'] = df.datetime.dt.date
    df.set_index('datetime', inplace=True)
    LOGGER.debug("Determining daily maximum wind speed record")
    dfmax = df.loc[df.groupby(['date'])[variable].idxmax().dropna()]
    dfdata = pd.DataFrame(columns=['v1', 'v2', 'r1', 'r2', 'category'],
                          index=dfmax.index)
    for idx, row in dfmax.iterrows():
        startdt = idx - timedelta(hours=2)
        enddt = idx + timedelta(hours=2)
        maxgust = row['windgust']
        v1 = df.loc[startdt:idx]['windgust'].mean()
        v2 = df.loc[idx:enddt]['windgust'].mean()
        r1 = maxgust/v1
        r2 = maxgust/v2
        if r1 < 2.0 and r2 < 2.0:
            category = 'synoptic'
        else:
            category = 'convective'
        dfdata.loc[idx] = [v1, v2, r1, r2, category]

    LOGGER.debug("Joining other observations to daily maximum wind data")
    dfmax = dfmax.join(dfdata)
    return dfmax


def dt(*args):
    """
    Convert args to `datetime`
    """
    return datetime(*args)


start()
