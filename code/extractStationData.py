"""
Extract time history of weather observations from 1-minute observational
weather station data, centred around daily maximum gust events.

We extract the daily maximum wind gust (including all the data variables
for that time) to a separate file. For all gusts > 90 km/h, we extract
additional data, that being the full time history for 1 hour either side
of the peak gust. If there is over 20% missing data in any of core variables
the data is then discarded (we won't use that event for subsequent
analysis).

Because the process takes some time (>24 hours) for all available stations,
the code is set up to record if an input file has been processed. This info
is stored in a text file (recording the file name, MD5 hash, date/time of
modification) and if attributes are not matched, the input file is processed.
You can turn this off in the configuration file with the `ExcludePastProcessed`
option set to `False`.

Processed files can be archived once processed. Set the `ArchiveWhenProcessed`
option to `True` and define `ArchiveDir` as a writable path.


To run:

python extractStationData.py -c extract_daily_1minmax.ini



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
import seaborn as sns
from prov.model import ProvDocument
from metpy.calc import wind_components
from metpy.units import units
import matplotlib
matplotlib.use("Agg")  # noqa E402
import matplotlib.pyplot as plt
import warnings

from process import pAlreadyProcessed, pWriteProcessedFile, pArchiveFile, pInit
from files import flStartLog, flGetStat, flSize
from stndata import ONEMINUTESTNNAMES, ONEMINUTEDTYPE, ONEMINUTENAMES


warnings.simplefilter('ignore', RuntimeWarning)
pd.set_option('mode.chained_assignment', None)
sns.set_style('whitegrid')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

LOGGER = logging.getLogger()
PATTERN = re.compile(r".*Data_(\d{6}).*\.txt")
STNFILE = re.compile(r".*StnDet.*\.txt")
TZ = {"QLD": 10, "NSW": 10, "VIC": 10,
      "TAS": 10, "SA": 9.5, "NT": 9.5,
      "WA": 8, "ANT": 0}

prov = ProvDocument()
prov.set_default_namespace("")
prov.add_namespace('prov', 'http://www.w3.org/ns/prov#')
prov.add_namespace('xsd', 'http://www.w3.org/2001/XMLSchema#')
prov.add_namespace('foaf', 'http://xmlns.com/foaf/0.1/')
prov.add_namespace('void', 'http://vocab.deri.ie/void#')
prov.add_namespace('dcterms', 'http://purl.org/dc/terms/')
provlabel = ":stormDataExtraction"
provtitle = "Storm data extraction"


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

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
        os.mkdir(pjoin(outputDir, 'dailymax'))
        os.mkdir(pjoin(outputDir, 'dailymean'))
        os.mkdir(pjoin(outputDir, 'plots'))
        os.mkdir(pjoin(outputDir, 'events'))
        os.mkdir(pjoin(outputDir, 'results'))

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
            if processFile(f, config):
                LOGGER.info(f"Successfully processed {f}")
                pWriteProcessedFile(f)
                if archiveWhenProcessed:
                    pArchiveFile(f)
                elif deleteWhenProcessed:
                    os.unlink(f)


def processFile(filename: str, config) -> bool:
    """
    process a file and store output in the given output directory

    :param str filename: path to a station data file
    :param config:  `ConfigParser` object
    """

    global g_stations
    outputFormat = config.get('Output', 'Format', fallback='pickle')
    outputDir = config.get('Output', 'Path')
    threshold = config.getfloat('Input', 'Threshold')
    ext = "pkl" if outputFormat == 'pickle' else 'csv'
    outfunc = "to_pickle" if outputFormat == 'pickle' else 'to_csv'
    LOGGER.info(f"Loading data from {filename}")
    LOGGER.debug(f"Data will be written to {outputDir}")
    LOGGER.debug(f"Using {outputFormat} for output data")
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

        basename = f"{stnNum:06d}.{ext}"  # os.path.basename(filename)
        dfmax, dfmean, eventdf = extractDailyMax(filename, stnState, stnName,
                                                 stnNum, 'windgust',
                                                 threshold=threshold)
        if dfmax is None:
            LOGGER.warning(f"No data returned for {filename}")
        else:
            LOGGER.debug(f"Writing data to {pjoin(outputDir, 'dailymax', basename)}")  # noqa: E501
            getattr(dfmax, outfunc)(pjoin(outputDir, 'dailymax', basename))
            getattr(dfmean, outfunc)(pjoin(outputDir, 'dailymean', basename))

            dfmax.to_pickle(pjoin(outputDir, 'dailymax', basename))
            dfmean.to_pickle(pjoin(outputDir, 'dailymean', basename))
            if eventdf is not None:
                LOGGER.debug(f"Writing data to {pjoin(outputDir, 'events', basename)}")  # noqa: E501
                getattr(eventdf, outfunc)(pjoin(outputDir, 'events', basename))

                eventdf.to_pickle(pjoin(outputDir, 'events', basename))
        rc = True
    return rc


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


def wdir_diff(wd1, wd2):
    """
    Calculate change in wind direction. This always returns a positive
    value, which is the minimum change in direction.

    :param wd: array of wind direction values
    :type wd: `np.ndarray`
    :return: Array of changes in wind direction
    :rtype: `np.ndarray`
    """

    diff1 = (wd1 - wd2) % 360
    diff2 = (wd2 - wd1) % 360
    res = np.minimum(diff1, diff2)
    return res


def extractDailyMax(filename, stnState, stnName, stnNum,
                    variable='windgust', threshold=90.):
    """
    Extract daily maximum value of `variable` from 1-minute observation records
    contained in `filename`

    :param filename: str, path object or file-like object
    :param stnState: str, station State (for determining local time zone)
    :param str variable: the variable to extract daily maximum values,
        default "windgust"
    :param float threshold: Threshold value of `variable` for additional
        data extraction - default 90.0

    :returns: `pandas.DataFrame`s

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
        return None, None
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

    dfdata = pd.DataFrame(columns=['gustratio', 'pretemp', 'posttemp',
                                   'temprise', 'tempdrop', 'dirrate',
                                   'prsdrop', 'prsrise', 'emergence'],
                          index=dfmax.index)
    LOGGER.debug("Calculating other obs around daily max wind gust")

    frames = []

    for idx in df.groupby(['date'])['windgust'].idxmax().dropna():
        startdt = idx - timedelta(hours=1)
        enddt = idx + timedelta(hours=1)
        maxgust = df.loc[idx]['windgust']
        wspd = df.loc[startdt: enddt]['windspd']
        meangust = df.loc[startdt: enddt]['windgust'].mean()
        pretemp = df.loc[startdt: idx]['temp'].mean()
        posttemp = df.loc[idx: enddt]['temp'].mean()
        tempchange = (df.loc[startdt: enddt]
                      .rolling('1800s')['temp']
                      .apply(lambda x: x[-1] - x[0])
                      .agg(['min', 'max']))
        maxtempchange = tempchange['max']
        mintempchange = tempchange['min']
        prschange = (df.loc[startdt: enddt]
                     .rolling('1800s')['stnp']
                     .apply(lambda x: x[-1] - x[0])
                     .agg(['min', 'max']))
        prsdrop = prschange['min']
        prsrise = prschange['max']
        dirchange = (df.loc[startdt: enddt]
                     .rolling('1800s')['winddir']
                     .apply(lambda x: wdir_diff(x[-1], x[0]))
                     .max())

        # Emergence: ratio of highest gust to average of next 10 highest
        emerg = maxgust / df.loc[startdt: enddt]['windgust'].nlargest(11)[1:].mean()  # noqa: E501
        dfdata.loc[idx] = [maxgust/meangust, pretemp, posttemp, maxtempchange,
                           mintempchange, dirchange, prsdrop, prsrise, emerg]

        if maxgust > threshold:
            LOGGER.info(("Extracting additional data for gust event on "
                         f"{datetime.strftime(idx, '%Y-%m-%d')}"))
            deltavals = df.loc[startdt: enddt].index - idx
            gustdf = df.loc[startdt: enddt].set_index(deltavals)
            pct = ((gustdf.isnull() | gustdf.isna()).sum() * 100 / gustdf.index.size)  # noqa: E501
            qccols = ['windgust', 'temp', 'stnp', 'winddir', 'dewpoint']
            if np.any(pct[qccols] > 20.0):
                LOGGER.info(("Missing values found in gust, temperature,"
                             "pressure, dewpoint or wind direction data"))
                continue

            gustdf['tempanom'] = gustdf['temp'] - gustdf['temp'].mean()
            gustdf['stnpanom'] = gustdf['stnp'] - gustdf['stnp'].mean()
            gustdf['dpanom'] = gustdf['dewpoint'] - gustdf['dewpoint'].mean()
            dirchange = (gustdf.rolling('1800s')['winddir']
                               .apply(lambda x: wdir_diff(x[-1], x[0])))
            gustdf['windchange'] = dirchange

            u, v, uanom, vanom = windComponents(gustdf['windspd'].values,
                                                gustdf['winddir'].values)
            gustdf['u'] = u
            gustdf['v'] = v
            gustdf['uanom'] = uanom
            gustdf['vanom'] = vanom
            x = (gustdf.index/60/1_000_000_000)
            newdf = gustdf[['windgust', 'temp', 'tempanom',
                            'windspd', 'winddir', 'windchange',
                            'u', 'v', 'uanom', 'vanom',
                            'stnp', 'stnpanom', 'rainfall', 'rh',
                            'dewpoint', 'dpanom']]
            newdf['tdiff'] = x.values.astype(float)
            newdf['date'] = datetime.strftime(idx, "%Y-%m-%d")
            newdf['time'] = df.loc[startdt: enddt].index.values
            frames.append(newdf)

    LOGGER.debug("Joining other observations to daily maximum wind data")
    dfmax = dfmax.join(dfdata)

    LOGGER.debug("Determining daily mean values")
    dfstats = df.groupby(['date']).aggregate({
        variable: ['mean', 'std', 'max'],
        'windspd': ['mean', 'std', 'max'],
        'temp': ['mean', 'min', 'max'],
        'temp1max': ['max'],
        'temp1min': ['min'],
        'wbtemp': ['mean', 'max'],
        'dewpoint': ['mean', 'max']})
    dfstats.columns = dfstats.columns.map('_'.join)
    if len(frames) > 0:
        eventdf = pd.concat(frames)
        return dfmax, dfstats, eventdf
    else:
        return dfmax, dfstats, None


def windComponents(windspd, winddir):
    """
    Calculate wind components and anomalies around the mean value

    :param windspd: wind speed data (in units of km/h)
    :type windspd: `pd.Series` or `np.array`
    :param winddir: wind direction data
    :type winddir: `pd.Series` or `np.array`

    :returns: `np.array` of U and V components (east and north)
    """

    u, v = wind_components(
        windspd * units.kph,
        winddir * units.deg
    )
    uanom = u.magnitude - np.nanmean(u.magnitude)
    vanom = v.magnitude - np.nanmean(v.magnitude)
    return u.magnitude, v.magnitude, uanom, vanom


def dt(*args):
    """
    Convert args to `datetime`
    """
    return datetime(*args)


start()
