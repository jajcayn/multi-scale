"""
created on July 23, 2014

@author: Nikola Jajcay
"""

from ecmwfapi import ECMWFDataServer
 
server = ECMWFDataServer()
 
server.retrieve({
    "stream" : "oper",
    "levtype" : "sfc",
    "param" : "167.128", ## https://badc.nerc.ac.uk/data/ecmwf-e40/params.html
    "dataset" : "interim", ## era40, interim, era20c
    "step" : "0",
    "grid" : "2.5/2.5",
    "time" : "00/06/12/18", ## daily
    "date" : "1980-01-01/to/2016-12-31",
    "area" : "75/-40/25/80", ## north/west/south/east
    "type" : "an", ## an for analysis, fc for forecast
    "class" : "ei", ## e4 for era40, ei for interim
    "format" : "netcdf",
    "padding" : "0",
    "target" : "ERAinterim.temp.EU.nc" ## filename
})

