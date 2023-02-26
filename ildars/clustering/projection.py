# Based on Rico Gießlers implementation in Mathematica

import numpy as np
import scipy as sp


# Threshold for detecting, whether a given arc is on a given hemisphere
# Threshold is directly taken from Rico Gießlers code, assuming the flag for
# 12 hemispheres to always be true
ARC_ON_HEMISPHERE_THRESHOLD = 0.7944


# Entry point of projection based clustering. Takes as input all measurements
# (ReflectedSignal class here)
def compute_reflection_clusters(reflected_signals):
    gnom_proj = compute_gnomonic_projection(reflected_signals)
    return None
    # TODO: Call the right functions from here
    # Step 1: Compute map circular segments to unit sphere
    # Step 2: Compute gnomonic projection for mapping (arcs)
    # Step 3: Compute clusters using the improved intersections algorithm


def compute_gnomonic_projection(reflected_signals):
    hemisphere_centers = get_12_hemispheres()
    arcs = compute_arcs(reflected_signals)
    lines = []
    for hemipoint in hemisphere_centers:
        hemilines = []
        for arc in arcs:
            arc_hemi_pos = find_arc_on_hemisphere(arc, hemipoint)
            if arc_hemi_pos is not None:
                hemilines.append(arc_hemi_pos)
        lines.append(hemilines)
    return lines


# get 12 random points for hemispheres
def get_12_hemispheres():
    vectors = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 1, 1]),
        np.array([1, 1, -1]),
        np.array([1, -1, -1]),
    ]
    # normalize all vectors
    vectors = [vec / np.linalg.norm(vec) for vec in vectors]

    # randomly rotate all vectors
    rotation = sp.spatial.transform.Rotation.random()
    rotated_vectors = rotation.apply(vectors)
    return np.append(rotated_vectors, -rotated_vectors, axis=0)


def compute_arcs(reflected_signals):
    arcs = []
    for sig in reflected_signals:
        v = sig.direct_signal.direction
        w = sig.direction
        delta = sig.delta
        p1 = (delta / 2) * w
        p2 = delta * ((w - v) / np.linalg.norm(w - v) ** 2)
        arcs.append((p1 / np.linalg.norm(p1), p2 / np.linalg.norm(p2)))
    return arcs


def find_arc_on_hemisphere(arc, hemipoint):
    pass


def FindGroups(lines, lineids):
    for line1 in range(Length(lines)):
        for line2 in range(line1 + 1, Length(lines)):
            if DoIntersect(lines[line1], lines[line2]):
                pairs.append(UndirectedEdge[line1, line2])
                idpairs.append(UndirectedEdge[lineids[line1], lineids[line2]])

                groups = ConnectedComponents[Graph[pairs]]
                idgroups = ConnectedComponents[Graph[idpairs]]
                for t in range(Length[groups]):
                    listfill = []
                    for r in range(Length[groups[t]]):
                        listfill.append(lines[groups[t][r]])
                        if Length[listfill] != 0:
                            walls.append(listfill)

    return (walls, idgroups)


# OLD FUNCTIONS


# new implementation of rotation using 3d rotation matrix
def rotate(theta, vec):
    # Vectors of the matrix
    rot_x_3d = [
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ]
    rot_y_3d = [
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ]
    rot_z_3d = [
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ]
    vec_arr = [vec[0], vec[1], vec[2]]

    vec_x_axis = np.matmul(vec_arr, rot_x_3d)
    vec_y_axis = np.matmul(vec_arr, rot_y_3d)
    vec_z_axis = np.matmul(vec_arr, rot_z_3d)

    return (vec_x_axis, vec_y_axis, vec_z_axis)


def find_center_point(points):
    ang = angleBetweenTwoVecor(points[[1]], points[[2]])
    co = np.cos(ang) / 2
    centerpoint = {0, 0, 0} + (
        (preprocessing.normalize[(points[1] + points[2]) / 2 - {0, 0, 0}])
    ) / co
    return (centerpoint, co)


def Calc_lat_long(points):
    y = points[[2]]
    radius = math.sqrt(x**2 + y**2 + z**2)
    lat = np.ArcSin(z / radius)
    lon = np.ArcTan(x, y)
    return (
        int(latitude / math.pi * 180),
        int((longitude) / math.pi * 180),
        int(radius),
    )


def cartesian_pts(points):
    lat = points[1]
    lon = points[2]
    radius = points[3]
    return (
        int(
            radius * np.cos(radianToDegree(lat)) * np.cos(radianToDegree(lon))
        ),
        int(
            radius * np.cos(radianToDegree(lat)) * np.sin(radianToDegree(lon))
        ),
        int(radius * np.sin(radianToDegree(lat))),
    )


def getCosC(point, mappingpoint):
    return math.sin((radianToDegree(mappingpoint[1]))) * math.sin(
        (radianToDegree(point[1]))
    ) + math.cos((radianToDegree(mappingpoint[1]))) * math.cos(
        radianToDegree(point[1])
    ) * math.cos(
        radianToDegree(point[2] - mappingpoint[2])
    )


def mapping_2d(v):
    euc = math.sqrt(v[1] ** 2 + v[2] ** 2)
    return (v[1] / euc, v[2] / euc)


# Explanation TODO; Function by Rico Gießler
def mapping_3d(v):
    euc = math.sqrt(v[1] ** 2 + v[2] ** 2 + v[3] ** 2)
    return (v[1] / euc, v[2] / euc, v[3] / euc)


# Explanation TODO; Function by Rico Gießler


def MapToUnitCircleWithWall(L):
    localL = []
    for i in range(len(L)):
        localL.append(
            Mapping(L[i, 1]),
            Mapping(L[i, 2]),
            Mapping(L[i, 3]),
            Mapping(L[i, 4]),
        )
        return localL


def GetVector(points):
    return (points[1], points[2] - points[1])


# Explanation TODO; Function by Rico Gießler
def GetVectors(lines):
    for i in range(1, len(lines)):
        vectors.append(GetVector(lines[[i]]))
    return vectors


def ComputeClosestPoint2D(points):
    m = len(points)
    n = len(points[[1], [1]])
    G = []
    d = []
    for i in range(1, m):
        for j in range(1, n):
            row[j] = 1
            row[i + n] = -points[i, 2, j]
            G.append(row)
            d.append(points[i, 1, j])

    return (G, d)
