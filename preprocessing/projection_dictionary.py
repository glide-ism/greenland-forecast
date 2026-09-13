"""Project CRS for every Greenland domain.

EPSG:3413 (NSIDC Sea Ice Polar Stereographic North, 70 N / -45 E, WGS84) is
the projection of BedMachine, the MEaSUREs/ITS_LIVE velocity products, ATL15,
ArcticDEM and the ISMIP6/ISMIP7 standard grids, so every builder regrids onto
it and most sources need no rotation of vector fields.
"""
from pyproj import CRS

crs = CRS('EPSG:3413')
