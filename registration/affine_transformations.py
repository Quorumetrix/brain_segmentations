



import numpy as np



def generate_affine_matrix_from_anchoring(ox, oy, oz, ux, uy, uz, vx, vy, vz, width, height):
    # Create the transformation matrix according to the article
    transformation_matrix = np.array([
        [ux, vx, 0, ox],
        [uy, vy, 0, oy],
        [uz, vz, 1, oz],
        [0, 0, 0, 1]
    ])

    # Create a scaling matrix to account for image width and height
    scaling_matrix = np.array([
        [1/width, 0, 0, 0],
        [0, 1/height, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Combine the transformation matrix and scaling matrix
    affine_matrix = np.matmul(transformation_matrix, scaling_matrix)

    return affine_matrix

