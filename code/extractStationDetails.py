"""
extractStationDetails.py - extract station information from collection of files

A PROV (provenance) file is written to the output folder, setting out the vaious entities created, sources, and the associations between them. See https://www.w3.org/TR/prov-primer/ for more details.


To run:

python extractStationData.py -c extract_station_data.ini

"""

import os
import re
import sys
import time
import glob
import argparse
import getpass
import logging
from os.path import join as pjoin
from datetime import datetime, timedelta
from configparser import ConfigParser, ExtendedInterpolation
import pandas as pd
import geopandas as gpd
import numpy as np
from prov.model import ProvDocument
from metpy.calc import wind_components
from metpy.units import units

import warnings

from process import pAlreadyProcessed, pWriteProcessedFile, pArchiveFile, pInit
from files import flStartLog, flGetStat, flSize, flGitRepository
from files import flModDate, flPathTime
from stndata import ONEMINUTESTNNAMES


warnings.simplefilter("ignore", RuntimeWarning)
pd.set_option("mode.chained_assignment", None)


LOGGER = logging.getLogger()
PATTERN = re.compile(r".*Data_(\d{6}).*\.txt")
STNFILE = re.compile(r".*StnDet.*\.txt")
TZ = {
    "QLD": 10,
    "NSW": 10,
    "VIC": 10,
    "TAS": 10,
    "SA": 9.5,
    "NT": 9.5,
    "WA": 8,
    "ANT": 0,
}

DATEFMT = "%Y-%m-%d %H:%M:%S"

prov = ProvDocument()
prov.set_default_namespace("")
prov.add_namespace("prov", "http://www.w3.org/ns/prov#")
prov.add_namespace("xsd", "http://www.w3.org/2001/XMLSchema#")
prov.add_namespace("foaf", "http://xmlns.com/foaf/0.1/")
prov.add_namespace("void", "http://vocab.deri.ie/void#")
prov.add_namespace("dcterms", "http://purl.org/dc/terms/")
provlabel = ":stationDataExtraction"
provtitle = "Station details extraction"


def start():
    """
    Parse command line arguments, initiate processing module (for tracking
    processed files) and start the main loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Configuration file")
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
    args = parser.parse_args()

    configFile = args.config_file
    verbose = args.verbose
    config = ConfigParser(allow_no_value=True, interpolation=ExtendedInterpolation())
    config.optionxform = str
    config.read(configFile)
    config.configFile = configFile

    pInit(configFile)
    main(config, verbose)


def main(config, verbose=False):
    """
    Start logger and call the loop to process source files.

    :param config: `ConfigParser` object with configuration loaded
    :param boolean verbose: If `True`, print logging messages to STDOUT

    """
    logfile = config.get("Logging", "LogFile")
    loglevel = config.get("Logging", "LogLevel", fallback="INFO")
    verbose = config.getboolean("Logging", "Verbose", fallback=verbose)
    datestamp = config.getboolean("Logging", "Datestamp", fallback=False)
    LOGGER = flStartLog(logfile, loglevel, verbose, datestamp)
    outputDir = config.get("Output", "Path", fallback="")
    starttime = datetime.now().strftime(DATEFMT)
    commit, tag, dt, url = flGitRepository(sys.argv[0])

    prov.agent(
        sys.argv[0],
        {
            "dcterms:type": "prov:SoftwareAgent",
            "prov:Revision": commit,
            "prov:tag": tag,
            "dcterms:date": dt,
            "prov:url": url,
        },
    )

    # We use the current user as the primary agent:
    prov.agent(f":{getpass.getuser()}", {"prov:type": "prov:Person"})

    prov.agent(
        "GeoscienceAustralia",
        {"prov:type": "prov:Organisation", "foaf:name": "Geoscience Australia"},
    )

    prov.agent(
        "BureauOfMeteorology",
        {
            "prov:type": "prov:Organization",
            "foaf:name": "Bureau of Meteorology Climate Data Services",
            "foaf:mbox": "climatedata@bom.gov.au",
        },
    )

    configent = prov.entity(
        ":configurationFile",
        {
            "dcterms:title": "Configuration file",
            "dcterms:type": "foaf:Document",
            "dcterms:format": "Text file",
            "prov:atLocation": os.path.basename(config.configFile),
        },
    )

    ListAllFiles(config)
    processStationFiles(config)

    endtime = datetime.now().strftime(DATEFMT)
    extractionact = prov.activity(
        provlabel,
        starttime,
        endtime,
        {"dcterms:title": provtitle, "dcterms:type": "void:Dataset"},
    )
    prov.actedOnBehalfOf(extractionact, f":{getpass.getuser()}")
    prov.actedOnBehalfOf(f":{getpass.getuser()}", "GeoscienceAustralia")
    prov.used(provlabel, configent)
    prov.wasAssociatedWith(extractionact, sys.argv[0])

    prov.serialize(pjoin(outputDir, "station_details.xml"), format="xml")

    for key in g_files.keys():
        LOGGER.info(f"Processed {len(g_files[key])} {key} files")
    LOGGER.info("Completed")


def ListAllFiles(config):
    """
    For each item in the 'Categories' section of the configuration file, load
    the specification (glob) for the files, then pass to `expandFileSpecs`

    :param config: `ConfigParser` object
    """
    global g_files
    g_files = {}
    categories = config.items("Categories")
    for idx, category in categories:
        specs = []
        items = config.items(category)
        for k, v in items:
            if v == "":
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

    origindir = config.get(
        category, "OriginDir", fallback=config.get("Defaults", "OriginDir")
    )
    dirmtime = flPathTime(origindir)
    specent = prov.collection(
        f":{spec}",
        {
            "dcterms:type": "prov:Collection",
            "dcterms:title": category,
            "prov:atLocation": origindir,
            "prov:GeneratedAt": dirmtime,
        },
    )
    prov.used(provlabel, specent)
    prov.wasAttributedTo(specent, "BureauOfMeteorology")
    specpath = pjoin(origindir, spec)
    files = glob.glob(specpath)
    entities = []
    LOGGER.info(f"{len(files)} {spec} files to be processed")
    for file in files:
        if os.stat(file).st_size > 0:
            if file not in g_files[category]:
                g_files[category].append(file)
                entities.append(
                    prov.entity(
                        f":{os.path.basename(file)}",
                        {
                            "prov:atLocation": origindir,
                            "dcterms:created": flModDate(file),
                        },
                    )
                )
    for entity in entities:
        prov.hadMember(specent, entity)


def expandFileSpecs(config, specs, category):
    """
    Expand a collection of file specifications

    :param config: `ConfigParser` object
    :param list specs: list of file specifications to expand
    :param str category: A category that has a section in the source
        configuration file
    """
    for spec in specs:
        expandFileSpec(config, spec, category)


def processStationFiles(config):
    """
    Process all the station files to populate a global station file DataFrame

    This produces two files: a GeoJSON file and a text file. The contents are
    almost identical - the GeoJSON file includes geometry and can be used to
    populate a Feature Class in an Esri File Geodatabase.

    :param config: `ConfigParser` object
    """

    global g_files
    global g_stations
    global LOGGER
    unknownDir = config.get("Defaults", "UnknownDir")
    deleteWhenProcessed = config.getboolean(
        "Files", "DeleteWhenProcessed", fallback=False
    )
    archiveWhenProcessed = config.getboolean(
        "Files", "ArchiveWhenProcessed", fallback=True
    )
    outputDir = config.get("Output", "Path", fallback=unknownDir)
    LOGGER.debug(f"DeleteWhenProcessed: {deleteWhenProcessed}")
    LOGGER.debug(f"Output directory: {outputDir}")
    if not os.path.exists(unknownDir):
        os.mkdir(unknownDir)

    stnlist = []
    category = "StationFiles"

    for f in g_files[category]:
        LOGGER.info(f"Processing {f}")
        directory, fname, md5sum, moddate = flGetStat(f)
        if pAlreadyProcessed(directory, fname, "md5sum", md5sum):
            LOGGER.info(f"Already processed {f}")
        else:
            df = getStationList(f)
            LOGGER.debug(f"Read station data from {f}")
            stnlist.append(df)
            pWriteProcessedFile(f)
            if archiveWhenProcessed:
                pArchiveFile(f)
            elif deleteWhenProcessed:
                os.unlink(f)

    g_stations = pd.concat(stnlist)

    LOGGER.debug("Creating GeoDataFrame for station data")
    gdf_stations = gpd.GeoDataFrame(
        data=g_stations,
        geometry=gpd.points_from_xy(g_stations.stnLon, g_stations.stnLat),
    )

    gdf_stations.set_crs(epsg=7844, inplace=True)
    geojsonfile = pjoin(outputDir, "StationDetails.geojson")
    gdf_stations.to_file(geojsonfile, driver="GeoJSON")

    # Provenance:
    geostnlist = prov.entity(
        ":GeospatialStationData",
        {
            "dcterms:type": "void:dataset",
            "dcterms:description": "Geospatial station information",
            "prov:atLocation": geojsonfile,
            "prov:GeneratedAt": datetime.now().strftime(DATEFMT),
            "dcterms:format": "GeoJSON",
        },
    )
    prov.wasGeneratedBy(geostnlist, provlabel, time=datetime.now().strftime(DATEFMT))

    txtfile = pjoin(outputDir, "StationDetails.txt")
    gdf_stations.reset_index()[ONEMINUTESTNNAMES].to_csv(txtfile, index=False)
    txtstnlist = prov.entity(
        ":TextStationData",
        {
            "dcterms:type": "void:dataset",
            "dcterms:description": "Text format station information",
            "prov:atLocation": txtfile,
            "prov:GeneratedAt": datetime.now().strftime(DATEFMT),
            "dcterms:format": "csv",
        },
    )
    prov.wasGeneratedBy(txtstnlist, provlabel, time=datetime.now().strftime(DATEFMT))


def getStationList(stnfile: str) -> pd.DataFrame:
    """
    Extract a list of stations from a station file

    :param str stnfile: Path to a station file

    :returns: :class:`pd.DataFrame`
    """
    LOGGER.debug(f"Retrieving list of stations from {stnfile}")
    df = pd.read_csv(
        stnfile,
        sep=",",
        index_col="stnNum",
        names=ONEMINUTESTNNAMES,
        keep_default_na=False,
        converters={"stnName": str.strip, "stnState": str.strip},
    )
    LOGGER.debug(f"There are {len(df)} stations")
    return df


start()
