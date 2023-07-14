import numpy as np
import pandas as pd

def rigidez_portico_matrix(E, A, L, I):

    K = np.zeros(shape=(6, 6))
    K[0, 0] = 1
    K[0, 3] = -1
    K[3, 0] = -1
    K[3, 3] = 1

    K *= (E * A) / L

    K[1, :] = np.array([0, 12, 6*L, 0, -12, 6*L]) * ((E * I)/(L**3))

    K[2, :] = np.array([0, 6*L, 4*(L**2), 0, -6*L, 2*(L**2)]
                       ) * ((E * I)/(L**3))

    K[4, :] = np.array([0, -12, -6*L, 0, 12, 6*L]) * ((E * I)/(L**3))

    K[5, :] = np.array([0, 6*L, 2*(L**2), 0, -6*L,
                       4*(L**2)]) * ((E * I)/(L**3))

    return K


def portico_rotation_matrix(theta):
    T = np.zeros(shape=(6, 6))
    theta = np.radians(theta)
    T[0, :] = [np.cos(theta), np.sin(theta), 0, 0, 0, 0]
    T[1, :] = [- np.sin(theta), np.cos(theta), 0, 0, 0, 0]
    T[2, 2] = 1
    T[3, :] = [0, 0, 0, np.cos(theta), np.sin(theta), 0]
    T[4, :] = [0, 0, 0, -np.sin(theta), np.cos(theta), 0]
    T[5, 5] = 1

    return T


def rigidez_trelica_matrix(E, A, L):
    K = np.zeros(shape=(6, 6))

    K[0, :] = [1, 0, 0, -1, 0, 0]
    K[3, :] = [1, 0, 0, 1, 0, 0]

    K *= (E * A)/L

    return K


def trelica_rotation_matrix(theta):
    T = np.zeros(shape=(6, 6))
    theta = np.radians(theta)
    T[0, :] = [np.cos(theta), np.sin(theta), 0, 0, 0, 0]
    T[1, :] = [- np.sin(theta), np.cos(theta), 0, 0, 0, 0]
    T[2, 2] = 1
    T[3, :] = [0, 0, 0, np.cos(theta), np.sin(theta), 0]
    T[4, :] = [0, 0, 0, -np.sin(theta), np.cos(theta), 0]
    T[5, 5] = 1

    return T


def generate_dataframe_from_matrix(local_matrix, nodes):

    column_names = []
    index_names = []
    for node in nodes:
        u_dof = f"{node}_u"
        v_dof = f"{node}_v"
        phi_dof = f"{node}_phi"

        column_names += [u_dof, v_dof, phi_dof]
        index_names += [u_dof, v_dof, phi_dof]

    matrix_df = pd.DataFrame(
        data=local_matrix, columns=column_names, index=index_names)

    return matrix_df


def discretize_portico(start_node, end_node, delta_x):
    x_start, y_start = start_node
    x_end, y_end = end_node

    length = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
    num_elements = int(length / delta_x)

    node_list = []

    dx = (x_end - x_start) / num_elements
    dy = (y_end - y_start) / num_elements

    for i in range(num_elements + 1):
        x = x_start + i * dx
        y = y_start + i * dy
        node_list.append([x, y])

    return np.array(node_list)


def generate_global_matrix(matrixes_list):
    global_matrix = pd.concat(matrixes_list).groupby(level=0).sum()
    return global_matrix

class Portico():

stiff_1 = rigidez_trelica_matrix(E=5, A=4, L=4)
t_inclination_1 = trelica_rotation_matrix(60)
rotated_1 = np.matmul(np.matmul(np.transpose(
    t_inclination_1), stiff_1), t_inclination_1)


stiff_2 = rigidez_portico_matrix(E=3, A=2, L=2, I=0.1)
t_inclination_2 = portico_rotation_matrix(0)

rotated_2 = np.matmul(np.matmul(np.transpose(
    t_inclination_2), stiff_2), t_inclination_2)


df1 = generate_dataframe_from_matrix(rotated_2, [0, 1])
df2 = generate_dataframe_from_matrix(rotated_1, [1, 7])
