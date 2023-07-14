import numpy as np


def rigidez_portico_matrix(E, A, L, I):

    K = np.zeros(shape=(6, 6))
    K[0, 0] = 1
    K[0, 3] = -1
    K[3, 0] = -1
    K[3, 3] = 1

    K *= (E * A) / L

    K[1, :] = [0, 12, 6*L, 0, -12, 6*L] * (E * I)/(L**3)

    K[2, :] = [0, 6*L, 4*(L**2), 0, -6*L, 2*(L**2)] * (E * I)/(L**3)

    K[4, :] = [0, -12, -6*L, 0, 12, 6*L] * (E * I)/(L**3)

    K[5, :] = [0, 6*L, 2*(L**2), 0, -6*L,  4*(L**2)]

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
    K = np.zeros(shape=(4, 4))

    K[0, :] = [1, 0, -1, 0]
    K[2, :] = [1, 0, 1, 0]

    K *= (E * A)/L

    return K


def trelica_rotation_matrix(theta):
    T = np.zeros(shape=(4, 4))
    theta = np.radians(theta)

    T[0, :] = [np.cos(theta), np.sin(theta), 0, 0]
    T[1, :] = [-np.sin(theta), np.cos(theta), 0, 0]
    T[2, :] = [0, 0, np.cos(theta), np.sin(theta)]
    T[3, :] = [0, 0, -np.sin(theta), np.cos(theta)]

    return np.round(T)


def generate_global_stiffness_matrix(nodes_list, local_matrixes):

    KG = np.zeros(shape=(2*len(nodes_list), 2*len(nodes_list)))

    for matrix, component_nodes in local_matrixes:
        matrix_end = len(matrix[0])

        initial_index = 2*component_nodes[0]
        final_index = matrix_end + initial_index
        print(KG[initial_index: final_index,
                 initial_index: final_index])

        KG[initial_index: final_index,
           initial_index: final_index] += matrix

    return KG


stiff_1 = rigidez_trelica_matrix(E=10, A=2, L=1)
t_inclination_1 = trelica_rotation_matrix(0)
rotated_1 = np.matmul(np.matmul(np.transpose(
    t_inclination_1), stiff_1), t_inclination_1)


stiff_2 = rigidez_trelica_matrix(E=10, A=1, L=1)
t_inclination_2 = trelica_rotation_matrix(90)
rotated_2 = np.matmul(np.matmul(np.transpose(
    t_inclination_2), stiff_2), t_inclination_2)



print(generate_global_stiffness_matrix([0, 1, 2], [
      (rotated_1, [0, 1]), (rotated_2, [1, 2])]))
