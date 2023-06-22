@echo off
title Extract daily maxima from 1-minute observations
CALL conda.bat activate process

python %CD%\extractStationData.py -c %CD%\extract_daily_1minmax.ini