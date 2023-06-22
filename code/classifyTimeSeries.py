"""
Train and run an automatic classification algorithm for determining the likely
phenomena that generated an severe wind gust.

Based on the temporal evolution of wind gusts, temperature, dew point and
station pressure, we can classify storms into synoptic or convective storms
(and then further sub-categories). We follow an approach similar to Cook (2023)
where a set of high-quality stations are used to develop a training dataset,
which is then used to train a machine-learning algorithm that can classify the
full set of events extracted.


"""

from os.path import join as pjoin
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import patheffects
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sktime.classification.kernel_based import RocketClassifier

from stndata import ONEMINUTESTNNAMES
from files import flStartLog

np.random.seed(1000)

# Following can be put into command line args or config file:
BASEDIR = r"X:\georisk\HaRIA_B_Wind\data\derived\obs\1-minute\events"
OUTPUTPATH = pjoin(r"X:\georisk\HaRIA_B_Wind\data\derived\obs\1-minute\events-60", "results")  # noqa
LOGGER = flStartLog(pjoin(OUTPUTPATH, "classifyTimeSeries.log"), "INFO", verbose=True)  # noqa
stndf = pd.read_csv(pjoin(BASEDIR, 'hqstations.csv'), index_col="stnNum")
eventFile = pjoin(BASEDIR, "CA_20230518_Hobart.csv")

# This is the visually classified training dataset:
stormdf = pd.read_csv(eventFile, usecols=[1, 2, 3], parse_dates=['date'],
                      dtype={'stnNum': int,
                             'stormType': 'category'})

# eventFile = pjoin(BASEDIR, "NA_all.csv")
# stormdf = pd.read_csv(eventFile, usecols=[2, 3, 4], parse_dates=['date'],
#                 dtype={'stnNum': float,
#                        'stormType': 'category'},
#                 converters={'stnNum': lambda s: int(float(s.strip() or 0))})

stormdf.set_index(['stnNum', 'date'], inplace=True)
nevents = len(stormdf)
LOGGER.info(f"{nevents} visually classified storms loaded from {eventFile}")
# To demonstrate the performance of the algorithm, we take a random selection
# of 200 storms to test against:
test_storms = stormdf.sample(200)
train_storms = stormdf.drop(test_storms.index)
ntrain = len(train_storms)


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
    for var in vars:
        df[var] = (df[var]
                   .interpolate(method='linear')
                   .fillna(method='bfill')
                   )

    df['stnNum'] = stnNum
    df.reset_index(inplace=True)
    df.set_index(['stnNum', 'date'], inplace=True)
    return df


def plotEvent(df: pd.DataFrame, stormType: str):
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
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    lnt = axt.plot(df.tdiff, df.tempanom, label=r"Temperature anomaly [$^o$C]",
                   color='r', marker='^', markerfacecolor="None",
                   lw=2, path_effects=[pe], zorder=1,
                   markevery=5)
    lnd = axt.plot(df.tdiff, df.dpanom, color='orangered', marker='.',
                   markerfacecolor="None", lw=1, path_effects=[pe],
                   zorder=1, markevery=5, label=r"Dew point anomaly [$^o$C]")
    lnp = axp.plot(df.tdiff, df.stnpanom, color='purple', lw=2,
                   path_effects=[pe], ls='--',
                   label='Station pressure anomaly [hPa]')
    lnw = ax.plot(df.tdiff, df.windgust, label="Gust wind speed [km/h]",
                  lw=3, path_effects=[pe], markerfacecolor="None", zorder=100)

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

    gmin, gmax = ax.get_ylim()
    pmin, pmax = axp.get_ylim()
    tmin, tmax = axt.get_ylim()
    ax.set_ylim((0, max(gmax, 100)))
    ax.set_xlabel("Time from gust peak(minutes)")
    axp.set_ylim((min(-2.0, pmin), max(pmax, 2.0)))
    axt.set_ylim((min(-2.0, tmin), max(tmax, 2.0)))

    ax.grid(True)
    axt.grid(False)
    axp.grid(False)

    lns = lnw + lnt + lnd + lnp
    labs = [ln.get_label() for ln in lns]
    ax.set_title(stormType)
    ax.legend(lns, labs)
    plt.text(1.0, -0.05, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=ax.transAxes, ha='right', va='top')
    plt.savefig(pjoin(OUTPUTPATH, f"{stormType}.png"), bbox_inches='tight')


# Load all the events into a single dataframe. We'll then pick out the events
# based on whether they are in the training set or the test set, using the
# index from the storm classification data:
LOGGER.info("Creating dataframe with all visually classified storm data")
dflist = []
for stn in stndf.index:
    df = loadData(stn, BASEDIR)
    dflist.append(df)

alldf = pd.concat(dflist)
alldf['idx'] = alldf.index



# Split into a preliminary training and test dataset:
LOGGER.info("Splitting the visually classified data into test and train sets")
eventdf_train = alldf.loc[train_storms.index]
eventdf_test = alldf.loc[test_storms.index]

vars = ['windgust', 'tempanom', 'stnpanom', 'dpanom']
nvars = len(vars)
# Apply a standard scaler (zero mean and unit variance):
# scaler = StandardScaler()
# eventdf_train[vars] = scaler.fit_transform(eventdf_train[vars].values)
# eventdf_test[vars] = scaler.transform(eventdf_test[vars].values)

X = eventdf_train.reset_index().set_index(['idx', 'tdiff'])[vars]
XX = np.moveaxis(X.values.reshape((ntrain, 121, nvars)), 1, -1)
y = np.array(train_storms['stormType'].values)

X_test = eventdf_test.reset_index().set_index(['idx', 'tdiff'])[vars]
XX_test = np.moveaxis(X_test.values.reshape((200, 121, nvars)), 1, -1)

# Here we use the full set of visually classified events for training
# the classifier:
fulltrain = (alldf.loc[stormdf.index]
             .reset_index()
             .set_index(['idx', 'tdiff'])[vars])
fulltrainarray = np.moveaxis(
    (fulltrain.values
     .reshape((len(stormdf), 121, nvars))), 1, -1)

# Create array of storm types from the visually classified data:
fully = np.array(list(
    stormdf.loc[fulltrain.reset_index()['idx'].unique()]['stormType'].values))

# First start with the training set:
LOGGER.info("Running the training set")
rocket = RocketClassifier(num_kernels=10000)
rocket.fit(XX, y)
y_pred = rocket.predict(XX_test)
results = pd.DataFrame(data={'prediction': y_pred,
                             'visual': test_storms['stormType']})
score = rocket.score(XX_test, test_storms['stormType'])
LOGGER.info(f"Accuracy of the classifier for the training set: {score}")

# Now run the classifier on the full event set:
LOGGER.info("Running classifier for all visually-classified events")
rocket = RocketClassifier(num_kernels=10000)
rocket.fit(fulltrainarray, fully)
newclass = rocket.predict(fulltrainarray)
results = pd.DataFrame(data={'prediction':newclass, 'visual':fully})
score = rocket.score(fulltrainarray, fully)
LOGGER.info(f"Accuracy of the classifier: {score}")
stormclasses = ['Synoptic storm', 'Synoptic front', 'Storm-burst',
                'Thunderstorm', 'Front up', 'Front down',
                'Spike', 'Unclassified']
(pd.crosstab(results['visual'], results['prediction'])
 .reindex(stormclasses)[stormclasses]
 .to_excel(pjoin(BASEDIR, 'events', 'crosstab.xlsx')))

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
        df = loadData(stn, r"X:\georisk\HaRIA_B_Wind\data\derived\obs\1-minute\events-60")  # noqa
    except FileNotFoundError:
        pass  # print(f"No data for station: {stn}")
    else:
        alldatadflist.append(df)

alldatadf = pd.concat(alldatadflist)
alldatadf['idx'] = alldatadf.index
allX = alldatadf.reset_index().set_index(['idx', 'tdiff'])[vars]

# Remove any events that have < 121 observations, or have missing
# data in any of the variables.
naidx = []
LOGGER.info("Removing storms with insufficient data:")
for ind, tmpdf in allX.groupby(level='idx'):
    if len(tmpdf) < 121:
        naidx.append(ind)
        LOGGER.info(f"{ind}, {len(tmpdf)}")
    if tmpdf.isna().sum().sum() > 0:
        # Found NAN values in the data (usually dew point)
        naidx.append(ind)
        LOGGER.info(f"{ind}, {len(tmpdf)}")

allXupdate = allX.drop(naidx, level='idx')
nstorms = int(len(allXupdate)/121)
vars = ['windgust', 'tempanom', 'stnpanom', 'dpanom']
nvars = len(vars)
allXX = np.moveaxis(allXupdate.values.reshape((nstorms, 121, nvars)), 1, -1)

LOGGER.info(f"Running the classifier for all {nstorms} events")
stormclass = rocket.predict(allXX)

outputstormdf = pd.DataFrame(data={'stormType': stormclass},
                             index=(allXupdate
                                    .index
                                    .get_level_values(0)
                                    .unique())
                             )

# Plot the storm counts
LOGGER.info("Plotting results")
outputstormdf.stormType.value_counts().plot(kind='bar')
plt.savefig(pjoin(OUTPUTPATH, "stormcounts.png"), bbox_inches='tight')

# Write the value counts to file
outputstormdf.stormType.value_counts().to_excel(
    pjoin(OUTPUTPATH, "stormcounts.xlsx"))

# Now we are going to plot the mean profile of the events:
allXupdate['idx'] = allXupdate.index.get_level_values(0)

for storm in stormclasses:
    LOGGER.info(f"Plotting mean profile for {storm} class events")
    stidx = outputstormdf[outputstormdf['stormType'] == storm].index
    stevents = allXupdate[allXupdate.index.get_level_values('idx').isin(stidx)]
    meanst = stevents.groupby('tdiff').mean().reset_index()
    plotEvent(meanst, storm)

outputFile = pjoin(OUTPUTPATH, 'stormclass.pkl')
LOGGER.info(f"Saving storm classification data to {outputFile}")
outputstormdf['stnNum'], outputstormdf['date'] = \
    zip(*outputstormdf.reset_index()['idx'])

outputstormdf.to_pickle(outputFile)
LOGGER.info("Completed")
