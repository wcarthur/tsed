import os
import re
from os.path import join as pjoin, basename, dirname
import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

dataPath = "X:/georisk/HaRIA_B_Wind/data/raw/from_bom/2019/Daily"
tempPath = "C:/WorkSpace/temp"
pattern = re.compile(".*StnDet.*\.txt")

filelist = os.listdir(dataPath)
for f in filelist:
    if pattern.match(f):
        stnfile = f

colnames = ["id", 'stnNum', 'rainfalldist', 'stnName', 'stnOpen', 'stnClose',
            'stnLat', 'stnLon', 'stnLoc', 'stnState', 'stnElev',
            'stnBarmoeterElev', 'stnWMOIndex', 'stnDataStartYear',
            'stnDataEndYear', 'pctComplete', 'pctY', 'pctN', 'pctW',
            'pctS', 'pctI']

datacols = ['id', 'stnNum',
            'YYYY', 'MM', 'DD',
            'windgust', 'windgustq', 'gustdir', 'gustdirq',
            'gusttime', 'gusttimeq']
usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

climatoldfnames = ["Filename", "Longitude", "Latitude", "Elevation",
                   "StationNumber", "StationName"]

print(f"Loading data from {f}")
df = pd.read_csv(pjoin(dataPath,stnfile), sep=',', index_col=False,
                 names=colnames, header=0, keep_default_na=False)
outdf = pd.DataFrame(columns = df.columns)
climatoldf = pd.DataFrame(columns=["Filename", "Longitude", "Latitude",
                                   "Elevation", "StationNumber",
                                   "StationName"])
for idx, row in df.iterrows():
    filename = f"DC02D_Data_{row.stnNum:06d}_999999999632559.txt"
    tmpdf = pd.read_csv(pjoin(dataPath, filename), sep=",", index_col=False,
                        header=0, names=datacols, usecols=usecols,
                        skipinitialspace=True)
    nrecords = tmpdf.windgust.notna().sum()
    if nrecords >= 365 * 5:
        outdf = outdf.append(row, ignore_index=True)
    if nrecords >= 365 * 10:
        climatoldf = climatoldf.append(
            pd.DataFrame([[pjoin(dataPath, filename),
                           row.stnLon, row.stnLat, row.stnElev,
                           row.stnNum, row.stnName]],
                           columns=climatoldfnames), ignore_index=True)
    else:
        print(f"Insufficient data for station {row.stnNum}")

climatoldf.to_csv(pjoin(tempPath, "stations.txt"), index=False)
gdf = gpd.GeoDataFrame(outdf,
                       geometry=gpd.points_from_xy(
                           outdf.stnLon,outdf.stnLat, crs="EPSG:7844"))
gdf.to_file(pjoin(tempPath, "daily-stationlist.shp"))
gdf.to_file(pjoin(tempPath, "daily-stationlist.json"), driver='GeoJSON')
gdf.drop(columns=['geometry']).to_csv(
    pjoin(tempPath, "daily-stationlist.csv"), index=False)

states = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')

ax = plt.axes(projection=ccrs.PlateCarree())
ax.figure.set_size_inches(15,10)
gdf.plot(ax=ax, marker='o', color='red', markersize=15, alpha=0.75,
         edgecolor='white', zorder=1)
ax.coastlines(resolution='10m')
ax.add_feature(states, edgecolor='0.15', linestyle='--')
gl = ax.gridlines(draw_labels=True, linestyle=":")
gl.top_labels = False
gl.right_labels = False
ax.set_extent([105, 160, -45, -5])
ax.set_title("AWS stations [>5 years data]")
ax.set_aspect('equal')
plt.savefig(pjoin(tempPath, "daily_aws_stations.png"), bbox_inches='tight')
