import numpy as np
import ildars

# "Small" number that the determinant is compared to
EPSILON = 0.000000001


def generate_measurements(receiver_position, sender_positions, room):
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
            room.vertices[face[0]],
            room.vertices[face[1]],
            room.vertices[face[2]],
        ]
        for face in room_triangle_indices
    ]
    for sender_position in sender_positions:
        # Compute direct signal direction and length
        direct_signal_direction = np.subtract(
            sender_position, receiver_position
        )
        direct_signal_length = np.linalg.norm(direct_signal_direction)
        direct_signal_direction = np.divide(
            direct_signal_direction, np.linalg.norm(direct_signal_direction)
        )
        direct_signal = ildars.DirectSignal(direct_signal_direction)
        # TODO: Check if the direct signal can actually be receiver or is
        # obstructed by a wall
        direct_signals.append(direct_signal)
        # compute reflection point for each triangle
        for triangle in room_triangles:
            edge1 = np.subtract(triangle[1], triangle[0])
            edge2 = np.subtract(triangle[2], triangle[0])
            normal = np.cross(edge1, edge2)
            line_direction = np.divide(normal, np.linalg.norm(normal))
            intersection_point = get_line_plane_intersection(
                receiver_position, line_direction, triangle[0], line_direction
            )
            if intersection_point is None:
                # Continue with next triangle, if no intersection exists
                continue
            mirror_point = np.add(
                receiver_position,
                np.multiply(
                    np.subtract(intersection_point, receiver_position), 2
                ),
            )
            # Compute and normalize reflection direction and o then intersect
            # it with walls
            reflection_direction = np.subtract(mirror_point, sender_position)
            reflection_direction = np.divide(
                reflection_direction, np.linalg.norm(reflection_direction)
            )
            reflection_point = get_line_triangle_intersection(
                sender_position, reflection_direction, triangle
            )
            if reflection_point is None:
                continue
            reflected_signal_direction = np.subtract(
                reflection_point, receiver_position
            )
            reflected_signal_length = np.linalg.norm(
                reflected_signal_direction
            ) + np.linalg.norm(np.subtract(sender_position, reflection_point))
            reflected_signal_direction = np.divide(
                reflected_signal_direction,
                np.linalg.norm(reflected_signal_direction),
            )
            reflected_signals.append(
                ildars.ReflectedSignal(
                    reflected_signal_direction,
                    direct_signal,
                    reflected_signal_length - direct_signal_length,
                    reflection_index,
                    np.array(sender_position),
                )
            )
            reflection_index += 1

    return (direct_signals, reflected_signals)


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
