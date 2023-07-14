import numpy as np
import pandas as pd


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











class Portico():
    
    def __init__(self, delta, start_node, end_node, theta, element_name):
        
        self.delta = delta
        self.stiffness_matrix_list = self.generate_local_stiffness_matrix()
        self.start_node = start_node
        self.end_node = end_node
        self.E = 210   # Parâmetro do ACO SAE 1045
        self.theta = theta
        self.element_name = element_name
    
    def generate_stiffness_matrix(self):
        # Discretiza o dominio de acordo com delta, gerando listas de nos que serao feitas
        nodes_list = self.discretize_portico(self.start_node, self.end_node)
        
        local_matrixes = []
        # Com a lista de nos, gera uma matriz local para cada elemento discretizado
        for element_nodes in nodes_list:
            
            # Calcula parametros que vao mudar com o elemento
            I, A = self.identify_element(self.element_name)
                  
            # Gera a matriz local e rotaciona de acordo com a necessidade
            stiff_matrix = self.rigidez_portico_matrix(self.E, A, self.delta, I)
            rotation_matrix = self.rotate_portico_matrix(self.theta)
            
            stiffness_matrix = np.matmul(np.matmul(np.transpose(
                rotation_matrix), stiff_matrix), rotation_matrix)
            
            df_matrix = self.generate_dataframe_from_matrix(stiffness_matrix, element_nodes)
            local_matrixes.append(df_matrix)

        # Usa todas as matrizes locais e gera uma global para o elemento de portico
        global_matrix = self.generate_global_matrix(local_matrixes)
        return global_matrix

    def generate_global_matrix(matrixes_list):
        global_matrix = pd.concat(matrixes_list).groupby(level=0).sum()
        return global_matrix
    def generate_dataframe_from_matrix(self, local_matrix, nodes):

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
    
    def rotate_portico_matrix(theta):
        T = np.zeros(shape=(6, 6))
        theta = np.radians(theta)
        T[0, :] = [np.cos(theta), np.sin(theta), 0, 0, 0, 0]
        T[1, :] = [- np.sin(theta), np.cos(theta), 0, 0, 0, 0]
        T[2, 2] = 1
        T[3, :] = [0, 0, 0, np.cos(theta), np.sin(theta), 0]
        T[4, :] = [0, 0, 0, -np.sin(theta), np.cos(theta), 0]
        T[5, 5] = 1

        return T

    def rigidez_portico_matrix(self, E, A, L, I):
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

    def discretize_portico(self, start_node, end_node):
        x_start, y_start = start_node
        x_end, y_end = end_node

        length = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
        num_elements = int(length / self.delta)

        node_list = []

        dx = (x_end - x_start) / num_elements
        dy = (y_end - y_start) / num_elements

        for i in range(num_elements + 1):
            x = x_start + i * dx
            y = y_start + i * dy
            node_list.append([x, y])

        return np.array(node_list)

