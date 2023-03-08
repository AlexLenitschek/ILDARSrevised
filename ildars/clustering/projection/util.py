# Collection of helper functions for gnomonic projection
import numpy as np


def normalize(v):
    return v / np.linalg.norm(v)


# get the angle between two vectors. Implementation taken from
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def get_angle(v1, v2):
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))


# get latitude and longituede in radians for given vector
def carth_to_lat_lon(v):
    lat = np.arcsin(v[2])
    lon = np.arctan2(v[0], v[1])
    return (lat, lon)


# TODO: what does this function acutally do geometrically?
def get_cos_c(v_ll, center_ll):
    v_lat = v_ll[0]
    v_lon = v_ll[1]
    center_lat = center_ll[0]
    center_lon = center_ll[1]
    return np.sin(center_lat) * np.sin(v_lat) + (
        np.cos(center_lat) * np.cos(v_lat) * np.cos(v_lon - center_lon)
    )
