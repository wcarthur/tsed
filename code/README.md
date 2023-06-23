# Thunderstorm event database

To classify events into convective or non-convective storms based on analysis of evolution of wind speed, temperature, station pressure and other meteorlogical variables.

This code uses `sktime` to perform a supervised classification of the time series of weather variables. A training event set is developed from visual classification of wind gust events over 90 km/h at high-quality stations. The full event set comprises all storm events with a maximum wind gust in excess of 60 km/h.

Quality flags are used to eliminate spurious gusts, but there still remain some anomalous events (instrument spikes).

Data from this analysis is used to create a geodatabase for publication and use in GIS applications.


## Data source

1-minute weather observations from all available weather stations across Australia.

Variables required:
- gust wind speed
- wind speed
- wind direction
- temperature
- dew point
- station pressure
- rainfall
- quality flags for all variables


## Requirements

- python 3.9
- jupyter
- numpy
- matplotlib
- pandas
- pytz
- seaborn
- sktime-all-extras
- gitpython
- geopandas
- cartopy
- scikit-learn
- metpy


## Installation

1. Build the conda environment
2. Install additional code from https://github.com/GeoscienceAustralia/nhi-pylib


## Process

1. extractStationData.py - extract all events from the raw data. This should be executed twice, initially with a threshold of 90 km/h and again with a threshold of 60 km/h. The outputs for each execution need to be stored in different folders
2. selectHQStations.ipynb
3. classifyGustEvents.py - classifies all daily maximum wind gusts using El Rafei et al. (2023)
4. ClassifyEvents.ipynb - interactive notebook to visually classify storms with maximum gust > 90 km/h at high-quality stations
5. classifyTimeSeries.py - use ML approach in sktime to classify all storm events (> 60 km/h)
6. convertStormCounts.py - convert classified storms to counts of storm type at each station
7. AnalyseClassifiedStorms.ipynb - interactive notebook to compare this classification against the El Rafei _et al._ gust classification
8. analyseStormEventTypes.ipynb - interactive notebook to examine the full classified storm event set, e.g. median and 99th percentile wind gusts for each storm type, seasonal distribution of storm type, comparison against other metrics (e.g. gust ratio, emergence). Still a work in progress.



## Products

_Link to eCat record for the data_

## Licence

_Add a licence_