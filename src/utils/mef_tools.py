import numpy as np
import pandas as pd


class Trelica():
    def __init__(self, A, E, L, theta, global_element_nodes):
        self.A = A
        self.E = E
        self.L = L
        self.theta = theta
        self.element_nodes = global_element_nodes

    def generate_trelica_matrix(self):
        stiff_matrix = self.rigidez_trelica_matrix(self.E, self.A, self.L)
        rotation_matrix = self.trelica_rotation_matrix(self.theta)

        stiffness_matrix = np.matmul(np.matmul(np.transpose(
            rotation_matrix), stiff_matrix), rotation_matrix)
        df_matrix = self.generate_dataframe_from_matrix(
            stiffness_matrix, self.element_nodes)

        return df_matrix

    def rigidez_trelica_matrix(self, E, A, L):
        K = np.zeros(shape=(4, 4))

        K[0, :] = [1, 0, -1, 0]
        K[2, :] = [1, 0, 1, 0]

        K *= (E * A)/L

        return K

    def trelica_rotation_matrix(self, theta):
        T = np.zeros(shape=(4, 4))
        theta = np.radians(theta)

        T[0, :] = [np.cos(theta), np.sin(theta), 0, 0]
        T[1, :] = [-np.sin(theta), np.cos(theta), 0, 0]
        T[2, :] = [0, 0, np.cos(theta), np.sin(theta)]
        T[3, :] = [0, 0, -np.sin(theta), np.cos(theta)]

        return T

    def generate_dataframe_from_matrix(self, local_matrix, nodes):

        column_names = []
        index_names = []
        for node in nodes:
            u_dof = f"{node}_u"
            v_dof = f"{node}_v"

            column_names += [u_dof, v_dof]
            index_names += [u_dof, v_dof]

        matrix_df = pd.DataFrame(
            data=local_matrix, columns=column_names, index=index_names)

        return matrix_df


class Portico():
    def __init__(self, delta, start_node, end_node, theta, element_name):

        self.delta = delta
        self.start_node = start_node
        self.end_node = end_node
        self.E = 210   # Parâmetro do ACO SAE 1045
        self.theta = theta
        self.element_name = element_name

    def generate_stiffness_matrix(self):
        # Discretiza o dominio de acordo com delta, gerando listas de nos que serao feitas
        nodes_list = self.discretize_portico(self.start_node, self.end_node)

        # Calcula parametros que vao mudar com o elemento
        I, A = self.identify_element(self.element_name)

        local_matrixes = []
        # Com a lista de nos, gera uma matriz local para cada elemento discretizado
        for element_nodes in nodes_list:

            # Gera a matriz local e rotaciona de acordo com a necessidade
            stiff_matrix = self.rigidez_portico_matrix(
                self.E, A, self.delta, I)
            rotation_matrix = self.rotate_portico_matrix(self.theta)

            stiffness_matrix = np.matmul(np.matmul(np.transpose(
                rotation_matrix), stiff_matrix), rotation_matrix)

            df_matrix = self.generate_dataframe_from_matrix(
                stiffness_matrix, element_nodes)
            local_matrixes.append(df_matrix)

        # Usa todas as matrizes locais e gera uma global para o elemento de portico
        global_matrix = self.generate_global_matrix(local_matrixes)
        return global_matrix

    def generate_global_matrix(self, matrixes_list):
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

    def rotate_portico_matrix(self, theta):
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

    def identify_element(self, element_name):
        # Return I, A
        if element_name == "A1":
            return (8.07332E-05, 0.000112626)
        elif element_name == "A2":
            return (3.26388E-09, 0.00009024)
        elif element_name == "A3":
            return (9.11458E-09, 0.000175)
        elif element_name == "A4":
            return (3.125E-10, 0.00015)
        else:
            raise Exception("Undefined element")


# General - Aqui vamos gerar os elementos de porticos e trelicas necessarios,
# definiremos os nos e as posicoes de cada portico e iremos juntar a matriz global aqui mesmo

# Definicao das variaveis globais
delta = 1E-02
alpha = 30
theta = 70

h = 400E-03
l = 700E-03
a = 550E-03
b = 150E-03
c = l/2
d = h/2
e = 30E-03
g = 180E-03
i = 250E-03

# Geracao dos porticos
# Portico A2
element_name = "A2"

portico_a2_horizontal_baixo = Portico(delta=delta, start_node=[
                                      e, 0], end_node=[e+l, 0], theta=0, element_name="A2")
portico_a2_horizontal_cima = Portico(delta=delta, start_node=[
    e, h], end_node=[e+l, h], theta=0, element_name="A2")

portico_a2_vertical_esquerda = Portico(delta=delta, start_node=[
e, 0], end_node=[e, h], theta=90, element_name="A2")

portico_a2_vertical_direita = Portico(delta=delta, start_node=[
                                      e+l, 0], end_node=[e+l, h], theta=90, element_name="A2")


# PORTICO A3
portico_a3_no1 = Portico(delta=delta, start_node=[
    0, 0], end_node=[e, 0], theta=0, element_name="A3")

portico_a3_no3 = Portico(delta=delta, start_node=[
    0, d], end_node=[e, d], theta=0, element_name="A3")

portico_a3_no5 = Portico(delta=delta, start_node=[
    e, h], end_node=[e, h+e], theta=90, element_name="A3")

portico_a3_no7 = Portico(delta=delta, start_node=[
    e+c, h], end_node=[e+c, h+c], theta=90, element_name="A3")

portico_a3_no6 = Portico(delta=delta, start_node=[
    e+l, h], end_node=[e+l, h+c], theta=90, element_name="A3")

portico_a3_no4 = Portico(delta=delta, start_node=[
    e+l, d], end_node=[e+l+e, d], theta=0, element_name="A3")

portico_a3_no2 = Portico(delta=delta, start_node=[
    e+l, 0], end_node=[e+l+e, 0], theta=0, element_name="A3")

# PORTICO A1

portico_a1_1_8 = Portico(delta=delta, start_node=[
    e, 0], end_node=[e+c, h], theta=np.degrees(np.arctan((h-0)/(e+c - e))), element_name="A1")


portico_a1_8_b = Portico(delta=delta, start_node=[
    e + c, h], end_node=[e+a, 0], theta=np.degrees(np.pi - np.arctan((c-a)/(h))), element_name="A1")

portico_a1_a_h = Portico(delta=delta, start_node=[
    e + a, 0], end_node=[e+a, h], theta=90, element_name="A1")


# TRELICA ESQUERDA


# TRELICA DIREITA


# TRELICA MOLA ESQUERDA


# TRELICA MOLA DIREITA

def generate_global_matrix(matrixes_list):
    global_matrix = pd.concat(matrixes_list).groupby(level=0).sum()
    return global_matrix
