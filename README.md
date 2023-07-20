# Thunderstorm event database

To classify events into convective or non-convective storms based on analysis of evolution of wind speed, temperature, station pressure and other meteorlogical variables.

This code uses `sktime`[^1] to perform a supervised classification of the time series of weather variables. A training event set is developed from visual classification of wind gust events over 90 km/h at high-quality stations. The full event set comprises all storm events with a maximum wind gust in excess of 60 km/h. The approach is based on previous concepts from Spassiani and Mason (2021)[^2] and Cook (2023)[^3].

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
3. Add the path to the additional code to the `PYTHONPATH` environment variable


## Process

1. `extractStationData.py` - extract all events from the raw data. This should be executed twice, initially with a threshold of 90 km/h and again with a threshold of 60 km/h. The outputs for each execution need to be stored in different folders. Users will need to check the path to the original source files (`OriginDir` in the configuration files.)
2. `selectHQStations.ipynb`
3. `classifyGustEvents.py` - classifies all daily maximum wind gusts using El Rafei et al. (2023)[^4]
4. `ClassifyEvents.ipynb` - interactive notebook to visually classify storms with maximum gust > 90 km/h at high-quality stations
5. `classifyTimeSeries.py` - use ML approach in sktime to classify all storm events (> 60 km/h)
6. `convertStormCounts.py` - convert classified storms to counts of storm type at each station
7. `AnalyseClassifiedStorms.ipynb` - interactive notebook to compare this classification against the El Rafei _et al._[^4] gust classification
8. `analyseStormEventTypes.ipynb` - interactive notebook to examine the full classified storm event set, e.g. median and 99th percentile wind gusts for each storm type, seasonal distribution of storm type, comparison against other metrics (e.g. gust ratio, emergence). Still a work in progress.



## Products

_Link to eCat record for the data_

## Licence

_Add a licence_


## References

[^1]: http://www.sktime.net/en/latest/index.html
[^2]: Spassiani, A. C., and M. S. Mason, 2021: Application of Self-organizing Maps to classify the meteorological origin of wind gusts in Australia. _Journal of Wind Engineering and Industrial Aerodynamics_, **210**, 104529, https://doi.org/10.1016/j.jweia.2021.104529.
[^3]: Cook, N. J., 2023: Automated classification of gust events in the contiguous USA. _Journal of Wind Engineering and Industrial Aerodynamics_, **234**, 105330, https://doi.org/10.1016/j.jweia.2023.105330.
[^4]: El Rafei, M., S. Sherwood, J. Evans, and A. Dowdy, 2023: Analysis and characterisation of extreme wind gust hazards in New South Wales, Australia. _Nat Hazards_, **117**, 875–895, https://doi.org/10.1007/s11069-023-05887-1.
