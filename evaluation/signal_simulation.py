import sys
sys.path.append ('../ILDARSrevised')
import random

import numpy as np

import ildars
import ildars.math_utils as util
from ildars.direct_signal import DirectSignal

# "Small" number that the determinant is compared to
EPSILON = 0.000000001
#center = np.array([0, 0, 0])   
# Define offset (distance from faces)
concert_offset = 0.1  # Change this value as needed
# Define offset for pyramidroom (distance from faces)
offset = 0.1  # Change this value as needed

###############################################################################################################################################
#THIS IS FOR PYRAMIDROOM - PUT OTHER ROOMS IN COMMENTS WHEN RUNNING PYRAMIDROOM
###############################################################################################################################################

# Define vertices
vertices = {
    'pA': np.array([1, 0.5, 0.5]),
    'pB': np.array([1, 0.5, -0.5]),
    'pC': np.array([-1, 0.5, -0.5]),
    'pD': np.array([-1, 0.5, 0.5]),
    'pE': np.array([0.5, -0.5, 0.25]),
    'pF': np.array([0.5, -0.5, -0.25]),
    'pG': np.array([-0.5, -0.5, -0.25]),
    'pH': np.array([-0.5, -0.5, 0.25]),

}

# Define faces based on connectivity (All anticlockwise)
faces = [
    ['pA', 'pB', 'pC', 'pD'],
    ['pA', 'pD', 'pH', 'pE'],
    ['pA', 'pE', 'pF', 'pB'],
    ['pG', 'pF', 'pE', 'pH'],
    ['pG', 'pH', 'pD', 'pC'],
    ['pG', 'pC', 'pB', 'pF'] 
]

# Access the minimum and maximum values for x, y, and z coordinates
min_x = np.min([vertex[0] for vertex in vertices.values()])
max_x = np.max([vertex[0] for vertex in vertices.values()])

min_y = np.min([vertex[1] for vertex in vertices.values()])
max_y = np.max([vertex[1] for vertex in vertices.values()])

min_z = np.min([vertex[2] for vertex in vertices.values()])
max_z = np.max([vertex[2] for vertex in vertices.values()])

# Calculate the center of the shape (mean of all vertices)
center = np.mean(list(vertices.values()), axis=0)

# Calculate normals for each face
face_normals = {}
for face_vertices in faces:
    vertices_list = [vertices[vertex] for vertex in face_vertices]
    vectors = [vertices_list[i + 1] - vertices_list[i] for i in range(len(vertices_list) - 1)]
    vectors.append(vertices_list[0] - vertices_list[-1])  # Connect last vertex to first
    
    normal = np.cross(vectors[0], vectors[1])
    face_normals[tuple(face_vertices)] = normal / np.linalg.norm(normal)

# Define offset (distance from faces)
offset = 0.1

# Check if a point is inside the shape considering the offset
def is_point_inside(point):
    for normal in face_normals.values():
        vec_to_center = center - point
        if np.dot(normal, vec_to_center) < -offset:
            return False
    return True

# Function to generate random points outside the offset distance from the faces
def generate_random_point():
    while True:
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        z = random.uniform(min_z, max_z)
        point = np.array([x, y, z])

        for face_vertices in faces:
            vertices_list = [vertices[vertex] for vertex in face_vertices]
            face_center = np.mean(vertices_list, axis=0)
            face_normal = np.cross(vertices_list[1] - vertices_list[0], vertices_list[2] - vertices_list[1])
            face_normal /= np.linalg.norm(face_normal)
            vec_to_face = face_center - point
            distance_to_face = np.dot(vec_to_face, face_normal)

            if distance_to_face < offset:
                break  # Point is inside the offset distance from this face
        else:
            #print(point)
            return point  # Point is outside the offset distance from all faces
        
###############################################################################################################################################
#SAME BUT FOR THE CONCERTHALL - PUT OTHER ROOMS IN COMMENTS WHEN RUNNING CONCERTHALL
###############################################################################################################################################

# # Define vertices
# concert_vertices = {
#     'cA': np.array([5.5, 0.8, 4.5]),
#     'cB': np.array([5.5, 0.8, -4.5]),
#     'cC': np.array([-4.5, 2.3, -1.2]),
#     'cD': np.array([-4.5, 2.3, 1.6]),
#     'cE': np.array([5.5, -4.0, 4.5]),
#     'cF': np.array([5.5, -4.0, -4.5]),
#     'cG': np.array([-4.5, 0.2, -1.2]),
#     'cH': np.array([-4.5, 0.2, 1.6]),

# }

# # Define faces based on connectivity (All anticlockwise)
# concert_faces = [
#     ['cA', 'cB', 'cC', 'cD'],
#     ['cA', 'cD', 'cH', 'cE'],
#     ['cA', 'cE', 'cF', 'cB'],
#     ['cG', 'cF', 'cE', 'cH'],
#     ['cG', 'cH', 'cD', 'cC'],
#     ['cG', 'cC', 'cB', 'cF'] 
# ]

# # Access the minimum and maximum values for x, y, and z coordinates
# concert_min_x = np.min([vertex[0] for vertex in concert_vertices.values()])
# concert_max_x = np.max([vertex[0] for vertex in concert_vertices.values()])

# concert_min_y = np.min([vertex[1] for vertex in concert_vertices.values()])
# concert_max_y = np.max([vertex[1] for vertex in concert_vertices.values()])

# concert_min_z = np.min([vertex[2] for vertex in concert_vertices.values()])
# concert_max_z = np.max([vertex[2] for vertex in concert_vertices.values()])

# # Calculate the center of the shape (mean of all vertices)
# concert_center = np.mean(list(concert_vertices.values()), axis=0)

# # Calculate normals for each face
# concert_face_normals = {}
# for concert_face_vertices in concert_faces:
#     concert_vertices_list = [concert_vertices[concert_vertex] for concert_vertex in concert_face_vertices]
#     concert_vectors = [concert_vertices_list[c + 1] - concert_vertices_list[c] for c in range(len(concert_vertices_list) - 1)]
#     concert_vectors.append(concert_vertices_list[0] - concert_vertices_list[-1])  # Connect last vertex to first
    
#     concert_normal = np.cross(concert_vectors[0], concert_vectors[1])
#     concert_face_normals[tuple(concert_face_vertices)] = concert_normal / np.linalg.norm(concert_normal)

# # Check if a point is inside the shape considering the offset
# def is_point_inside_concert(concert_point):
#     for concert_normal in concert_face_normals.values():
#         concert_vec_to_center = concert_point - concert_center
#         if np.dot(concert_normal, concert_vec_to_center) < -concert_offset:
#             return False
#     return True

# # Function to generate random points outside the offset distance from the faces
# def generate_random_point_in_concert():
#     while True:
#         concert_x = random.uniform(concert_min_x, concert_max_x)
#         concert_y = random.uniform(concert_min_y, concert_max_y)
#         concert_z = random.uniform(concert_min_z, concert_max_z)
#         concert_point = np.array([concert_x, concert_y, concert_z])

#         for concert_face_vertices in concert_faces:
#             concert_vertices_list = [concert_vertices[concert_vertex] for concert_vertex in concert_face_vertices]
#             concert_face_center = np.mean(concert_vertices_list, axis=0)
#             concert_face_normal = np.cross(concert_vertices_list[1] - concert_vertices_list[0], concert_vertices_list[2] - concert_vertices_list[1])
#             concert_face_normal /= np.linalg.norm(concert_face_normal)
#             concert_vec_to_face = concert_face_center - concert_point
#             concert_distance_to_face = np.dot(concert_vec_to_face, concert_face_normal)

#             if concert_distance_to_face < concert_offset:
#                 break  # Point is inside the offset distance from this face
#         else:
#             #print(concert_point)
#             return concert_point  # Point is outside the offset distance from all faces
        
###############################################################################################################################################

###############################################################################################################################################


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
    # sender_positions = [ # THIS IS FOR CUBE
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

    # Senderpositions for the new rooms

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
        generate_random_point()
        for j in range(num_senders)
    ]

    # sender_positions = [ # THIS IS FOR CONCERTHALL
    #     generate_random_point_in_concert()
    #     for j in range(num_senders)
    # ]

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