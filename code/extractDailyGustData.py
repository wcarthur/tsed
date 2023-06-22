import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
from configparser import ConfigParser, ExtendedInterpolation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns

from process import pAlreadyProcessed, pWriteProcessedFile, pArchiveFile, pInit
from files import flStartLog, flGetStat

import warnings
warnings.filterwarnings("ignore", pd.SettingWithCopyWarning)



PEAKDATAPATH = r"X:\georisk\HaRIA_B_Wind\data\derived\obs\1-minute\stats\dailymax"
FULLDATAPATH = r"X:\georisk\HaRIA_B_Wind\data\raw\from_bom\2022\1-minute"

# All time zones are based on local standard time. NO DST applied!
TZ = {"QLD":10, "NSW":10, "VIC":10, "TAS":10,
      "SA":9.5, "NT":9.5, "WA":8, "ANT":0}

COLNAMES = ['id', 'stnNum', 'YYYY', 'MM', 'DD', 'HH', 'MI',
            'YYYYl', 'MMl', 'DDl', 'HHl', 'MIl', 'rainfall', 'rainq',
            'rain_duration', 'temp', 'tempq', 'temp1max', 'temp1maxq',
            'temp1min', 'temp1minq', 'wbtemp', 'wbtempq', 'dewpoint',
            'dewpointq', 'rh', 'rhq', 'windspd', 'windspdq', 'windmin',
            'windminq', 'winddir', 'winddirq', 'windsd', 'windsdq',
            'windgust', 'windgustq', 'mslp', 'mslpq', 'stnp', 'stnpq', 'end']
COLDTYPE = {'id': str, 'stnNum': int, 'YYYY': int, "MM": int, "DD": int,
            "HH": int, "MI": int, 'YYYYl': int, "MMl": int, "DDl": int,
            "HHl": int, "MIl": int, 'rainfall': float, 'rainq': str,
            'rain_duration': float, 'temp': float, 'tempq': str,
            'temp1max': float, 'temp1maxq': str, 'temp1min': float,
            'temp1minq': str, 'wbtemp': float, 'wbtempq': str,
            'dewpoint': float, 'dewpointq': str, 'rh':float, 'rhq':str,
            'windspd':float, 'windspdq':str, 'windmin': float, 'windminq': str,
            'winddir': float, 'winddirq': str, 'windsd':float, 'windsdq':str,
            'windgust': float, 'windgustq': str, 'mslp':float, 'mslpq':str,
            'stnp': float, 'stnpq': str, 'end': str}

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

def main(config, verbose):
    """
    Main processing loop. Reads the list of stations, extracts wind events for
    each location and stores the data in a separate file, along with a plot of
    the data.

    :param config: `ConfigParser` object with configuration loaded
    :param boolean verbose: If `True`, print logging messages to STDOUT
    """

    logfile = config.get('Logging', 'LogFile')
    loglevel = config.get('Logging', 'LogLevel', fallback='INFO')
    verbose = config.getboolean('Logging', 'Verbose', fallback=verbose)
    datestamp = config.getboolean('Logging', 'Datestamp', fallback=False)
    LOGGER = flStartLog(logfile, loglevel, verbose, datestamp)

    loadStations()
    processStations()

def wdir_diff(wd1, wd2):
    """
    Calculate change in wind direction

    :param wd: array of wind direction values
    :type wd: `np.ndarray`
    :return: Array of changes in wind direction
    :rtype: `np.ndarray`
    """

    diff1 = (wd1 - wd2)% 360
    diff2 = (wd2 - wd1)% 360
    res = np.minimum(diff1, diff2)
    return res



def loadStations(config):
    """
    Load station details from the merged file

    :param config: `ConfigParser` object with configuration loaded
    """

    stnFile = config.get('Input', 'StationFile')

    LOGGER.debug(f"Retrieving list of stations from {stnFile}")
    colnames = ["id", 'stnNum', 'rainfalldist', 'stnName', 'stnOpen', 'stnClose',
            'stnLat', 'stnLon', 'stnLoc', 'stnState', 'stnElev', 'stnBarmoeterElev',
            'stnWMOIndex', 'stnDataStartYear', 'stnDataEndYear',
            'pctComplete', 'pctY', 'pctN', 'pctW', 'pctS', 'pctI', 'end']
    df = pd.read_csv(stnFile, sep=',', index_col=False, names=colnames,
                     keep_default_na=False,
                     converters={
                         'stnName': str.strip,
                         'stnState': str.strip
                         })
    LOGGER.debug(f"There are {len(df)} stations")
    return df

def processStations(config):

    global LOGGER
    unknownDir = config.get('Defaults', 'UnknownDir')
    originDir = config.get('Defaults', 'OriginDir')
    deleteWhenProcessed = config.getboolean('Files', 'DeleteWhenProcessed', fallback=False)
    archiveWhenProcessed = config.getboolean('Files', 'ArchiveWhenProcessed', fallback=True)
    outputDir = config.get('Output', 'Path', fallback=unknownDir)
    LOGGER.debug(f"Origin directory: {originDir}")
    LOGGER.debug(f"DeleteWhenProcessed: {deleteWhenProcessed}")
    LOGGER.debug(f"Output directory: {outputDir}")
    stndf =


def processStation(stnNum, threshold=90):

    df = pd.read_csv(os.path.join(PEAKDATAPATH, "HD01D_Data_072150_9999999910174132.txt"), index_col='datetime')
    peakdf = df[df.windgust > threshold]
    peakdf.set_index(pd.to_datetime(peakdf.index), inplace=True)

    filename = r"X:\georisk\HaRIA_B_Wind\data\raw\from_bom\2022\1-minute\HD01D_Data_072150_9999999910174132.txt"
    fulldf = pd.read_csv(filename, sep=',', index_col=False, dtype=COLDTYPE,
                         names=COLNAMES, header=0,
                         parse_dates={'datetime':[7, 8, 9, 10, 11]},
                         na_values=['####'], skipinitialspace=True)
    fulldf['datetime'] = pd.to_datetime(fulldf.datetime, format="%Y %m %d %H %M")
    fulldf['datetime'] = fulldf.datetime - timedelta(hours=10)
    fulldf['date'] = fulldf.datetime.dt.date
    fulldf = fulldf.set_index('datetime')

    for idx, row in peakdf.iterrows():
        startdt = idx - timedelta(hours=0.5)
        enddt = idx + timedelta(hours=0.5)
        deltavals = fulldf.loc[startdt : enddt].index - idx
        tmpdf = fulldf.loc[startdt : enddt].set_index(deltavals)

def plotGustEvent(df, dt):

    x = (df.index/60/1_000_000_000)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(x, df['windgust'], 'ko-', markerfacecolor='white')

    axs2 = ax.twinx()
    axs3 = ax.twinx()
    axs4 = ax.twinx()
    dirchange = (df.rolling('1800s')['winddir']
                   .apply(lambda x: wdir_diff(x[0], x[-1])))
    df['windchange'] = dirchange
    axs2.plot(x, dirchange, 'go', markersize=3, markerfacecolor='w')
    axs3.plot(x, df['temp'] - df['temp'].mean(), 'r')
    axs4.plot(x, df['mslp'] - df['mslp'].mean(), 'purple')
    axs3.spines[['right']].set_visible(False)
    axs4.spines[['right']].set_visible(False)
    axs3.tick_params(right=False, labelright=False)
    axs4.tick_params(right=False, labelright=False)

    axs4.text(0.95, 0.05, dt.strftime("%Y-%m-%d %H:%M"), transform=ax.transAxes, ha= 'right',
                bbox=dict(facecolor='white', edgecolor='black', pad=1.0, alpha=0.5), zorder=100)
    plt.savefig()