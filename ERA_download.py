"""
created on July 23, 2014

@author: Nikola Jajcay
"""

from ecmwfapi import ECMWFDataServer
 
server = ECMWFDataServer()
 
server.retrieve({
    "stream" : "oper",
    "levtype" : "pl",
    "param" : "60.128/129.128/130.128/131.128/132.128/133.128/135.128/138.128/155.128/157.128/203.128", ## https://badc.nerc.ac.uk/data/ecmwf-e40/params.html
    # "dataset" : "interim", ## era40, interim, era20c
    "levelist" : "100/300/500/800/850/900/925/950/1000",
    "step" : "0",
    # "grid" : "2.5/2.5",
    "time" : "12:00:00", ## daily
    "date" : "2017-04-04",
    # "area" : "75/-40/25/80", ## north/west/south/east
    "type" : "an", ## an for analysis, fc for forecast
    "class" : "od", ## e4 for era40, ei for interim
    # "format" : "netcdf",
    # "padding" : "0",
    "target" : "output", ## filename
    "expver" : 1
})

