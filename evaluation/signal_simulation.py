import sys
sys.path.append ('../ILDARSrevised')
import random

import numpy as np

import ildars
import ildars.math_utils as util
from ildars.direct_signal import DirectSignal

# "Small" number that the determinant is compared to
EPSILON = 0.000000001

# FOR PYRAMIDROOM Corner coordinates of the trapezoid
bottom_surface = np.array([(1, 0.5, 0.5), (1, 0.5, -0.5), (-1, 0.5, -0.5), (-1, 0.5, 0.5)])
top_surface = np.array([(0.5, -0.5, 0.25), (0.5, -0.5, -0.25), (-0.5, -0.5, -0.25), (-0.5, -0.5, 0.25)])
center = np.array([0, 0, 0])

# FOR PYRAMIDROOM Calculate smaller trapezoid for boundary
offset = 0.1  # Distance from the original trapezoid
boundary_bottom_surface = bottom_surface - ((bottom_surface - center) / np.linalg.norm(bottom_surface - center)) * offset
boundary_top_surface = top_surface - ((top_surface - center) / np.linalg.norm(top_surface - center)) * offset

# FOR PYRAMIDROOM Combine top and bottom surface coordinates for the boundary trapezoid
boundary_trapezoid = np.vstack((boundary_bottom_surface, boundary_top_surface))

# FOR PYRAMIDROOM Generate random sender positions inside the boundary trapezoid
def generate_random_point_in_boundary_trapezoid():
    while True:
        x = random.uniform(min(boundary_trapezoid[:, 0]), max(boundary_trapezoid[:, 0]))
        y = random.uniform(min(boundary_trapezoid[:, 1]), max(boundary_trapezoid[:, 1]))
        z = random.uniform(min(boundary_trapezoid[:, 2]), max(boundary_trapezoid[:, 2]))

        # Check if the point falls within the boundary trapezoid
        if any(
            [
                (boundary_trapezoid[i, 0] == x and boundary_trapezoid[i, 1] == y and boundary_trapezoid[i, 2] == z)
                for i in range(len(boundary_trapezoid))
            ]
        ):
            return np.array([x, y, z])

def generate_measurements(receiver_position, room, num_senders):
    direct_signals = []
    reflected_signals = []
    # For debugging: assign an index to each reflected signal
    reflection_index = 0

    # collect indices for faces
    room_triangle_indices = [room.meshes[key].faces for key in room.meshes]
    # Flatten list
    room_triangle_indices = [
        item for sublist in room_triangle_indices for item in sublist
    ]
    # Compute triangles
    room_triangles = [
        [
            np.subtract(room.vertices[face[0]], receiver_position),
            np.subtract(room.vertices[face[1]], receiver_position),
            np.subtract(room.vertices[face[2]], receiver_position),
        ]
        for face in room_triangle_indices
    ]
    # compute actual wall normal vectors
    wall_nv = [
        util.normalize(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        for tri in room_triangles
    ]

    # random sender positions
    # Following this are the old senderpositions for CUBE
    # sender_positions = [
    #     np.array(
    #         [
    #             random.uniform(-1, 1),
    #             random.uniform(-1, 1),
    #             random.uniform(-1, 1),
    #         ]
    #     )
    #     - receiver_position
    #     for i in range(num_senders)
    # ]

    # Senderpositions for the rooms

    # sender_positions = [ # THIS IS FOR TEST1ROOM
    #     np.array(
    #          [
    #              random.uniform(-2.1, 2.1),
    #              random.uniform(-0.9, 0.9),
    #              random.uniform(-1.6, 1.6),
    #          ]
    #     )
    #     - receiver_position
    #     for i in range(num_senders)
    # ]

    sender_positions = [ # THIS IS FOR PYRAMIDROOM
        generate_random_point_in_boundary_trapezoid()
        for _ in range(num_senders)
    ]

    for sender_position in sender_positions:
        # Compute direct signal direction and length
        direct_signal_direction = sender_position
        direct_signal_length = np.linalg.norm(direct_signal_direction)
        direct_signal_direction = util.normalize(direct_signal_direction)
        direct_signal = DirectSignal(direct_signal_direction, sender_position)
        # TODO: Check if the direct signal can actually be receiver or is
        # obstructed by a wall
        direct_signals.append(direct_signal)
        # compute reflection point for each triangle
        for triangle in room_triangles:
            edge1 = np.subtract(triangle[1], triangle[0])
            edge2 = np.subtract(triangle[2], triangle[0])
            normal = np.cross(edge1, edge2)
            line_direction = util.normalize(normal)
            intersection_point = get_line_plane_intersection(
                np.zeros(3), line_direction, triangle[0], line_direction
            )
            if intersection_point is None:
                # Continue with next triangle, if no intersection exists
                continue
            mirror_point = np.multiply(intersection_point, 2)
            # Compute and normalize reflection direction and o then intersect
            # it with walls
            reflection_direction = np.subtract(mirror_point, sender_position)
            reflection_direction = util.normalize(reflection_direction)
            reflection_point = get_line_triangle_intersection(
                sender_position, reflection_direction, triangle
            )
            if reflection_point is None:
                continue
            ref_signal_direction = reflection_point
            reflected_signal_length = np.linalg.norm(
                ref_signal_direction
            ) + np.linalg.norm(np.subtract(sender_position, reflection_point))
            ref_signal_direction = util.normalize(ref_signal_direction)
            new_ref_sig = ildars.ReflectedSignal(
                ref_signal_direction,
                direct_signal,
                reflected_signal_length - direct_signal_length,
                reflection_index,
            )
            reflected_signals.append(new_ref_sig)
            reflection_index += 1

    return (direct_signals, reflected_signals, wall_nv)


def get_line_triangle_intersection(line_point, line_direction, triangle):
    # Check if line l : line_point + x * normal intersects with our triangle,
    # using the Möller-Trumbore Algorithm as described in
    # https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    edge1 = np.subtract(triangle[1], triangle[0])
    edge2 = np.subtract(triangle[2], triangle[0])
    h = np.cross(line_direction, edge2)
    a = np.dot(edge1, h)
    if a > -EPSILON and a < EPSILON:
        # Line is parallel to the triangle. Continue with next line
        return None
    f = 1 / a
    s = np.subtract(line_point, triangle[0])
    u = np.multiply(f, np.dot(s, h))
    if u < 0 or u > 1:
        # line does not intersect with triangle
        return None
    q = np.cross(s, edge1)
    v = np.multiply(f, np.dot(line_direction, q))
    if v < 0 or np.add(u, v) > 1:
        # line does not intersect with triangle
        return None
    # now we know that an intersection exists. Next we compute the
    # intersection point
    t = np.multiply(f, np.dot(edge2, q))
    intersection_point = np.add(line_point, np.multiply(line_direction, t))
    return intersection_point


def get_line_plane_intersection(
    line_point, line_direction, plane_point, plane_normal
):
    # Compute the intersection of a line and an infinite plane based on
    # https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
    det = np.dot(plane_normal, line_direction)
    if abs(det) > EPSILON:
        w = np.subtract(line_point, plane_point)
        fac = -np.dot(plane_normal, w) / det
        u = np.multiply(line_direction, fac)
        return np.add(line_point, u)

    # line is parallel to the plane
    return None
