"""
Plot all events classified as spike events for quality control checking

"""
from os.path import join as pjoin
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # noqa E402
import matplotlib.pyplot as plt
from matplotlib import patheffects
from datetime import datetime


from stndata import ONEMINUTESTNNAMES
from files import flStartLog

LOGGER = flStartLog(r"..\output\plotSpikeEvents.log", "INFO", verbose=True)
OUTPUTPATH = pjoin(r"..\data\allevents", "results")
PLOTPATH = pjoin(r"..\data\allevents", "plots")


def loadData(stnNum: int, datapath: str) -> pd.DataFrame:
    """
    Load event data for a given station. Missing values are interpolated
    linearly - if values are missing at the start or end they are backfilled
    from the nearest valid value.

    This data has been extracted by `extractStationData.py`, and is stored in
    pickle files, so there should be no issues around type conversions, when
    used on the same machine.

    :param stnNum: BoM station number
    :type stnNum: int
    :return: DataFrame holding the data of all gust events for a station
    :rtype: `pd.DataFrame`
    """
    fname = pjoin(datapath, "events", f"{stnNum:06d}.pkl")
    LOGGER.debug(f"Loading event data from {fname}")
    df = pd.read_pickle(fname)
    df['date'] = pd.to_datetime(df['date'])
    vars = ['windgust', 'tempanom', 'stnpanom',
            'dpanom', 'windspd', 'uanom', 'vanom']

    # Temporarily switch off the interpolation and backfill
    # for var in vars:
    #     df[var] = (df[var]
    #                .interpolate(method='linear')
    #                .fillna(method='bfill')
    #                )

    df['stnNum'] = stnNum
    df.reset_index(inplace=True)
    df.set_index(['stnNum', 'date'], inplace=True)
    return df


def plotEvent(df: pd.DataFrame, stormType: str, stnNum: int, dtstr: str):
    """
    Plot the mean profile of an event

    :param df: DataFrame containing timeseries of temp, wind gust, etc.
    :type df: pd.DataFrame
    :param stormType: Name of the storm class
    :type stormType: str
    """
    pe = patheffects.withStroke(foreground="white", linewidth=5)

    fig, ax = plt.subplots(figsize=(12, 8))
    axt = ax.twinx()
    axp = ax.twinx()
    axd = ax.twinx()
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    lnt = axt.plot(df.tdiff, df.tempanom, label=r"Temperature anomaly [$^o$C]",
                   color='r', marker='^', markerfacecolor="None",
                   lw=1, zorder=1,
                   markevery=5)
    lnd = axt.plot(df.tdiff, df.dpanom, color='orangered', marker='.',
                   markerfacecolor="None", lw=1,
                   zorder=1, markevery=5, label=r"Dew point anomaly [$^o$C]")
    lnp = axp.plot(df.tdiff, df.stnpanom, color='purple', lw=1,
                   ls='--',
                   label='Station pressure anomaly [hPa]')
    lnw = ax.plot(df.tdiff, df.windgust, label="Gust wind speed [km/h]",
                  marker='o', lw=1, markerfacecolor="None", zorder=100)
    lnwd = axd.scatter(df.tdiff, df.winddir, color='g', marker='o',
                       label="Wind direction")

    axt.spines[['right']].set_color('r')
    axt.yaxis.label.set_color('r')
    axt.tick_params(axis='y', colors='r')
    axt.set_ylabel(r"Temperature/dewpoint anomaly [$^o$C]")

    ax.set_ylabel("Gust wind speed [km/h]")

    axp.spines[['right']].set_position(('axes', 1.075))
    axp.spines[['right']].set_color('purple')
    axp.yaxis.label.set_color('purple')
    axp.tick_params(axis='y', colors='purple')
    axp.set_ylabel("Pressure anomaly [hPa]")

    axd.spines[['right']].set_position(('axes', 1.15))
    axd.spines[['right']].set_color('g')
    axd.yaxis.label.set_color('g')
    axd.tick_params(axis='y', colors='g')
    axd.set_ylabel("Wind direction")

    gmin, gmax = ax.get_ylim()
    pmin, pmax = axp.get_ylim()
    tmin, tmax = axt.get_ylim()
    ax.set_ylim((0, max(gmax, 100)))
    ax.set_xlabel("Time from gust peak(minutes)")
    axp.set_ylim((min(-2.0, pmin), max(pmax, 2.0)))
    axt.set_ylim((min(-2.0, tmin), max(tmax, 2.0)))
    axd.set_ylim((0, 360))

    ax.grid(True)
    axt.grid(False)
    axp.grid(False)
    axd.grid(False)

    lns = lnw + lnt + lnd + lnp
    labs = [ln.get_label() for ln in lns]
    ax.set_title(f"{stnNum} - {dtstr}")
    ax.legend(lns, labs)
    plt.text(1.0, -0.05, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=ax.transAxes, ha='right', va='top')
    plt.savefig(pjoin(PLOTPATH, f"{stormType}.{stnNum}.{dtstr}.png"),
                bbox_inches='tight')
    plt.close(fig)


allstnfile = r"X:\georisk\HaRIA_B_Wind\data\raw\from_bom\2022\1-minute\HD01D_StationDetails.txt"  # noqa

allstndf = pd.read_csv(allstnfile, sep=',', index_col='stnNum',
                       names=ONEMINUTESTNNAMES,
                       keep_default_na=False,
                       converters={
                           'stnName': str.strip,
                           'stnState': str.strip
                       })

# Now load all events where the maximum wind gust exceeds 60 km/h
alldatadflist = []
LOGGER.info("Loading all events with maximum gust > 60 km/h")
for stn in allstndf.index:
    try:
        df = loadData(stn, r"..\data\allevents")
    except FileNotFoundError:
        pass  # print(f"No data for station: {stn}")
    else:
        alldatadflist.append(df)

alldatadf = pd.concat(alldatadflist)
alldatadf['idx'] = alldatadf.index

stormClassFile = pjoin(OUTPUTPATH, 'stormclass.pkl')
stormclassdf = pd.read_pickle(stormClassFile)

spikeEvents = stormclassdf[stormclassdf.stormType == 'Spike']
spidx = spikeEvents.index

spikeData = alldatadf[alldatadf['idx'].isin(spidx)]

for idx, data in spikeData.groupby('idx'):
    dtstr = idx[1].strftime('%Y%m%d')
    stnNum = idx[0]
    plotEvent(data, 'SPIKE', stnNum, dtstr)
