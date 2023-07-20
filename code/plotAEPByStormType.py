"""
Calculate and plot the annual exceedance probability curves for each station, separated by storm type, and a combined curve.

Storm classification completed in other scripts

"""

from os.path import join as pjoin, exists

import pandas as pd
import numpy as np
from scipy.stats import genpareto
from scipy.optimize import curve_fit
from statsmodels.nonparametric.kde import KDEUnivariate

import seaborn as sns
import matplotlib.pyplot as plt

from stndata import ONEMINUTESTNDTYPE, ONEMINUTESTNNAMES

BASEDIR = r"..\data\allevents"
OUTPUTPATH = pjoin(BASEDIR, "results")
stormdf = pd.read_pickle(pjoin(OUTPUTPATH, "stormclass.pkl"))
allstnfile = r"X:\georisk\HaRIA_B_Wind\data\raw\from_bom\2022\1-minute\HD01D_StationDetails.txt"
allstndf = pd.read_csv(allstnfile, sep=',', index_col='stnNum',
                       names=ONEMINUTESTNNAMES,
                       keep_default_na=False,
                       converters={
                            'stnName': str.strip,
                            'stnState': str.strip,
                            'stnDataStartYear': lambda s: int(float(s.strip() or 0)),
                            'stnDataEndYear': lambda s: int(float(s.strip() or 0))
                        })

# Calculate the time span that each station is open:
allstndf['timespan'] = allstndf['stnDataEndYear'] - allstndf['stnDataStartYear']

def loadDailyMaxData(stnNum: int) -> pd.DataFrame:
    """
    Load daily maximum wind gust data for a given station

    :param stnNum: Station identification number
    :type stnNum: int
    :return: `pd.DataFrame` containing daily maximum wind gust data for the station, along with other variables associated with the gust event.
    :rtype: pd.DataFrame
    """
    try:
        df = pd.read_pickle(pjoin(BASEDIR, 'dailymax', f"{stnNum:06d}.pkl"))
    except FileNotFoundError:
        print(f"No data for station {stnNum}")
    df = df.reset_index(drop=False).set_index(['stnNum', 'date'])
    return df

def plotDistribution(stnNum, tsgust, tparams, sygust, sparams, outputpath):
    """
    Plot a histogram distribution of the synoptic and convective wind gusts
    and the fitted GPD.

    :param stnNum: _description_
    :type stnNum: _type_
    :param tsgust: _description_
    :type tsgust: _type_
    :param tuple tparams: _description_
    :param sygust: _description_
    :type sygust: _type_
    :param tuple sparams: _description_
    :param str outputpath: _description_

    """
    x = np.arange(62, 200)
    fig, ax = plt.subplots(1, 1)
    sxi, smu, ssigma = sparams
    ax.plot(x, genpareto.pdf(x, sxi, loc=smu, scale=ssigma),
            label="Fitted GPD - SYN")
    ax.hist(sygust, density=True, alpha=0.5,
            label="Observations - SYN",
            bins=np.arange(60, 200, 2.5))
    ax.set_xlim(60, 10*(int(sygust.max()/10) + 1))

    txi, tmu, tsigma = tparams
    ax.plot(x, genpareto.pdf(x, txi, loc=tmu, scale=tsigma),
            color='k', label="Fitted GPD - TS")
    ax.hist(tsgust, density=True, color='g', alpha=0.5,
            label="Observations - TS",
            bins=np.arange(60, 200, 2.5))
    ax.set_xlim(60, 10*(int(tsgust.max()/10) + 1))
    ax.set_xlabel("Gust wind speed [km/h]")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.savefig(pjoin(outputpath, f"{stnNum}.hist.png"), bbox_inches='tight')


def plotReturnLevel(stnNum, tparams, sparams, intervals, outputpath):
    """
    Plot return level wind speeds for given set of GPD parameters

    :param int stnNum: Station number
    :param tuple tparams: tuple of GPD parameters for convective gusts
    :param tuple sparams: tuple of GPD parameters for synoptic gusts
    :param intervals: :class:`numpy.array` of return intervals to evaluate
    :param str outputpath: destination path for figures
    """
    # Work on the assumption that we have dailydata
    npyr = 365.25
    txi, tmu, tsigma = tparams
    sxi, smu, ssigma = sparams

    srate = float(len(swspd))/(allstndf.loc[stnNum]['timespan']*npyr)
    trate = float(len(twspd))/(allstndf.loc[stnNum]['timespan']*npyr)
    trl = tmu + (tsigma / txi) * (np.power(intervals * npyr * trate, txi) - 1.)
    srl = smu + (ssigma / sxi) * (np.power(intervals * npyr * srate, sxi) - 1.)

    fig, ax = plt.subplots(1, 1)
    ax.semilogx(intervals, trl, label="TS")
    ax.semilogx(intervals, srl, label='SYN')
    ax.legend(frameon=False)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')
    ax.set_xlabel("Return period [years]")
    ax.set_ylabel("Gust wind speed [km/h]")

alldflist = []
for stn in allstndf.index:
    if exists(pjoin(BASEDIR, 'dailymax', f"{stn:06d}.pkl")):
        df = loadDailyMaxData(stn)
        alldflist.append(df)
    else:
        print(f"No data for station {stn}")

alldf = pd.concat(alldflist)
alldf.reset_index(drop=False, inplace=True)
alldf['date'] = pd.to_datetime(alldf.date)
alldf.set_index(['stnNum', 'date'], inplace=True)

datadf = stormdf.join(alldf, on=['stnNum', 'date'], how='inner')

datadf.loc[(datadf.windgust>150) & (datadf.gustratio > 9.0) & (~datadf.stormType.isin(['Spike', 'Unclassified'])), 'stormType'] = 'Unclassified'

tsGust = datadf.loc[datadf.stormType.isin(['Thunderstorm', 'Front up', 'Front down'])]
synGust = datadf.loc[datadf.stormType.isin(['Synoptic storm', 'Synoptic front', 'Storm-burst'])]

tsgpdparams = pd.DataFrame(columns=['mu', 'xi', 'sigma'])
syngpdparams = pd.DataFrame(columns=['mu', 'xi', 'sigma'])

for stn, tmpdf in datadf.groupby('stnNum'):
    LOGGER.info(f"Processing {stn}")
    tsgust = tmpdf.loc[tmpdf.stormType.isin(['Thunderstorm', 'Front up', 'Front down'])].values
    sygust = tmpdf.loc[tmpdf.stormType.isin(['Synoptic storm', 'Synoptic front', 'Storm-burst'])].values
    txi, tmu, tsigma = genpareto.fit(np.sort(tsgust),fc=-0.1, loc=tsgust.min())
    tsgpdparams.loc[stn] = pd.Series({'mu': tmu, 'xi': txi, 'sigma': tsigma}, name=stn)
    sxi, smu, ssigma = genpareto.fit(np.sort(sygust),fc=-0.1, loc=sygust.min())
    syngpdparams.loc[stn] = pd.Series({'mu': smu, 'xi': sxi, 'sigma': ssigma}, name=stn)

    plotDistribution(stn, tsgust, sygust,
                     (txi, tmu, tsigma),
                     (sxi, smu, ssigma),
                     OUTPUTPATH)
    plotReturnLevel(stn, (txi, tmu, tsigma), (sxi, smu, ssigma), OUTPUTPATH)
    plotExceedance(stn, (txi, tmu, tsigma), (sxi, smu, ssigma), OUTPUTPATH)