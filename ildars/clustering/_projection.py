# Based on Rico Gießlers implementation in Mathematica

import numpy as np
import scipy as sp


# only used for debugging in early implementation phase
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
plt.style.use("seaborn-whitegrid")


# Threshold for detecting, whether a given arc is on a given hemisphere
# Threshold is directly taken from Rico Gießlers code, assuming the flag for
# 12 hemispheres to always be true
COS_C_THRESHOLD = 0.7944


# Entry point of projection based clustering. Takes as input all measurements
# (ReflectedSignal class here)
def compute_reflection_clusters(reflected_signals):
    gnom_proj = compute_gnomonic_projection(reflected_signals)

    # only fot debugging
    fig = plt.figure()
    cols = int(np.floor(np.sqrt(len(gnom_proj))))
    rows = int(np.ceil(len(gnom_proj) / cols))
    gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0)
    plt_rows = gs.subplots()
    i_hemisphere = 0
    for plt_row in plt_rows:
        for subplot in plt_row:
            for line in gnom_proj[i_hemisphere]:
                x = [line[0][0], line[1][0]]
                y = [line[0][1], line[1][1]]
                subplot.plot(x, y)
            if i_hemisphere < len(gnom_proj) - 1:
                i_hemisphere += 1
    plt.show()
    return None
    # TODO: Call the right functions from here
    # Step 1: Compute map circular segments to unit sphere
    # Step 2: Compute gnomonic projection for mapping (arcs)
    # Step 3: Compute clusters using the improved intersections algorithm


def compute_gnomonic_projection(reflected_signals):
    hemisphere_centers = get_12_hemispheres()
    arcs = compute_arcs(reflected_signals)
    lines = []
    for hemicenter in hemisphere_centers:
        hemilines = []
        for arc in arcs:
            arc_hemi_pos = find_arc_on_hemisphere(arc, hemicenter)
            if arc_hemi_pos is not None:
                hemilines.append(arc_hemi_pos)
        if len(hemilines) > 0:
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


def find_arc_on_hemisphere(arc, hemicenter):
    start_lat_lon = calc_lat_long(arc[0])
    end_lat_lon = calc_lat_long(arc[1])
    # arc_center = compute_arc_center(arc)
    arc_center = arc[0] + arc[1]
    arc_center /= np.linalg.norm(arc_center)
    center_lat_lon = calc_lat_long(arc_center)
    hemicenter_lat_lon = calc_lat_long(hemicenter)

    start_cos_c = get_cos_c(start_lat_lon, hemicenter_lat_lon)
    end_cos_c = get_cos_c(end_lat_lon, hemicenter_lat_lon)
    center_cos_c = get_cos_c(center_lat_lon, hemicenter_lat_lon)

    if (
        start_cos_c <= COS_C_THRESHOLD
        and end_cos_c <= COS_C_THRESHOLD
        and center_cos_c <= COS_C_THRESHOLD
    ):
        # arc is not present on hemisphere
        return None
    # Clip arcs to hemisphere if necessary
    if start_cos_c <= COS_C_THRESHOLD:
        print("pre adjust:", start_cos_c)
        start = clip_vector_to_hemisphere(arc[0], hemicenter)
        start_lat_lon = calc_lat_long(start)
        start_cos_c = get_cos_c(start_lat_lon, hemicenter_lat_lon)
        print("post adjust:", start_cos_c)
    if end_cos_c <= COS_C_THRESHOLD:
        end = clip_vector_to_hemisphere(arc[1], hemicenter)
        end_lat_lon = calc_lat_long(end)
        end_cos_c = get_cos_c(end_lat_lon, hemicenter_lat_lon)
    return (
        lat_lon_to_gnomonic_coordinates(
            start_lat_lon, hemicenter_lat_lon, start_cos_c
        ),
        lat_lon_to_gnomonic_coordinates(
            end_lat_lon, hemicenter_lat_lon, end_cos_c
        ),
    )


def clip_vector_to_hemisphere(vec, hemicenter):
    rot_ax = np.cross(vec, hemicenter)
    if not np.any(rot_ax):
        print("given vector is parallel or opposite of hemicenter")
        return vec
    posrotvec = sp.spatial.transform.Rotation.from_rotvec(
        rot_ax * COS_C_THRESHOLD
    )
    negrotvec = sp.spatial.transform.Rotation.from_rotvec(
        (-rot_ax) * COS_C_THRESHOLD
    )
    posrot = posrotvec.apply(vec)
    negrot = negrotvec.apply(vec)
    if angle_between(vec, posrot) < angle_between(vec, negrot):
        return posrot
    return negrot


def lat_lon_to_gnomonic_coordinates(ll_point, ll_hemi, cos_c_point):
    lat_hemi = ll_hemi[0]
    lon_hemi = ll_hemi[1]
    lat_point = ll_point[0]
    lon_point = ll_point[1]
    x = (
        1
        / cos_c_point
        * np.cos(np.radians(lat_point))
        * np.sin(np.radians(lon_point - lon_hemi))
    )
    y = (
        1
        / cos_c_point
        * (
            np.cos(np.radians(lat_hemi)) * np.sin(np.radians(lat_point))
            - np.sin(np.radians(lat_hemi))
            * np.cos(np.radians(lat_point))
            * np.cos(np.radians(lon_point - lon_hemi))
        )
    )
    return (x, y)


def compute_arc_center(arc):
    ang = angle_between(arc[0], arc[1])
    co = np.cos(ang / 2)
    if co == 0:
        print("found co of 0")
        return np.array([0, 0, 0])
    centerpoint = (arc[0] + arc[1]) / 2
    centerpoint /= np.linalg.norm(centerpoint)
    centerpoint *= 1 / co
    return centerpoint


# implementation taken from
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def angle_between(v1, v2):
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))


def calc_lat_long(point):
    x = point[0]
    y = point[1]
    z = point[2]
    lat = np.arcsin(z)
    lon = np.arctan(x / y)
    return int(lat / np.pi * 180), int((lon) / np.pi * 180)


def get_cos_c(point, mappingpoint):
    return (
        np.sin(np.radians(mappingpoint[0])) * np.sin(np.radians(point[0]))
    ) + (
        np.cos(np.radians(mappingpoint[0]))
        * np.cos(np.radians(point[0]))
        * np.cos(np.radians(point[1] - mappingpoint[1]))
    )


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
