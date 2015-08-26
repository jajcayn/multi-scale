"""
created on July 23, 2014

@author: Nikola Jajcay
"""

from ecmwfapi import ECMWFDataServer
 
server = ECMWFDataServer()
 
server.retrieve({
    "stream" : "oper",
    "levtype" : "sfc",
    "param" : "167.128",
    "dataset" : "era40", ## era40, interim
    "step" : "0",
    "grid" : "2.5/2.5",
    "time" : "00/06/12/18", ## daily
    "date" : "19580101/to/20131231",
    # "area" : "75/-40/25/80", ## north/west/south/east
    "type" : "an",
    "class" : "e4",
    "format" : "netcdf",
    "padding" : "0",
    "target" : "ERA40.t2m.nc" ## filename
})

