import numpy as np
# from scipy.linalg import eigh
# import matplotlib.pyplot as plt

# # Parâmetros da estrutura
# L = 1.0  # Comprimento total da estrutura
# dx_values = [0.01, 0.02]  # Valores de discretização para os elementos

# # Função para construir a matriz de rigidez global


# # Calcular frequências naturais e modos de vibração
# def calculate_modes_frequencies(K):
#     # Calcular autovalores e autovetores
#     eigenvalues, eigenvectors = eigh(K)

#     # Ordenar em ordem crescente
#     idx = np.argsort(eigenvalues)
#     eigenvalues = eigenvalues[idx]
#     eigenvectors = eigenvectors[:, idx]

#     return eigenvalues, eigenvectors

# # Plotar modos de vibração
# def plot_modes(eigenvectors, dx):
#     num_modes = eigenvectors.shape[1]

#     for i in range(num_modes):
#         plt.plot(eigenvectors[:, i], label='Modo {}'.format(i + 1))

#     plt.xlabel('Posição')
#     plt.ylabel('Deslocamento')
#     plt.title('Modos de Vibração')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Loop para diferentes valores de discretização
# for dx in dx_values:
#     K = build_global_stiffness_matrix(L, dx)
#     eigenvalues, eigenvectors = calculate_modes_frequencies(K)

#     # Converter autovalores para frequências em Hertz
#     frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

#     print('Discretização (dx = {}):'.format(dx))
#     for i in range(6):
#         print('Frequência {}: {:.2f} Hz'.format(i + 1, frequencies[i]))

#     if dx == min(dx_values):
#         plot_modes(eigenvectors, dx)



import matplotlib.pyplot as plt


class Beam:
    def __init__(self, young, inertia, rho, area, node, bar):
        self.young = young
        self.inertia = inertia
        self.rho = rho
        self.area = area
        self.node = node.astype(float)
        self.bar = bar.astype(int)

        self.dof = 2
        self.point_load = np.zeros_like(node)

        self.support = np.ones_like(node).astype(int)
        self.section = np.ones(len(bar))

        self.force = np.zeros([len(bar), 2 * self.dof])
        self.displacement = np.zeros([len(bar), 2 * self.dof])


# Structure Input
E: float = 10  # Pa
I: float = 500  # in^4
rho: float = 7860  # Kg/m^3
area: float = 10  # m^3
delta_x = 0.1  # 1cm

def create_beam(start_node, end_node, delta_x):
    x_start, y_start = start_node
    x_end, y_end = end_node

    length = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
    num_elements = int(length / delta_x)

    node_list = []
    element_list = []

    dx = (x_end - x_start) / num_elements
    dy = (y_end - y_start) / num_elements

    for i in range(num_elements + 1):
        x = x_start + i * dx
        y = y_start + i * dy
        node_list.append([x, y])

    for i in range(num_elements):
        element = [i, i + 1]
        element_list.append(element)

    return np.array(node_list), np.array(element_list)

def analysis(self, ss, si):
    ne = len(self.bar)
    d = self.node[self.bar[:, 1], :] - self.node[self.bar[:, 0], :]
    length = np.sqrt((d ** 2).sum(axis=1))

    # Form Structural Stiffness and Inertia
    k_matrix = np.zeros([2 * self.dof, 2 * self.dof])
    m_matrix = np.zeros([2 * self.dof, 2 * self.dof])
    k = np.zeros([ne, 2 * self.dof, 2 * self.dof])
    m = np.zeros([ne, 2 * self.dof, 2 * self.dof])

    for i in range(ne):
        # Generate DOF
        aux = self.dof * self.bar[i, :]
        index = np.r_[aux[0]:aux[0] + self.dof, aux[1]:aux[1] + self.dof]

        # Element Stiffness Matrix and Inertia Matrix
        l: float = length[i]
        k_matrix[0] = [12, 6 * l, -12, 6 * l]
        k_matrix[1] = [6 * l, 4 * l ** 2, -6 * l, 2 * l ** 2]
        k_matrix[2] = [-12, -6 * l, 12, -6 * l]
        k_matrix[3] = [6 * l, 2 * l ** 2, -6 * l, 4 * l ** 2]
        k[i] = self.young * self.inertia * k_matrix / l ** 3

        m_matrix[0] = [156, 22 * l, 59, -13 * l]
        m_matrix[1] = [22 * l, 4 * l ** 2, 13 * l, -3 * l ** 2]
        m_matrix[2] = [54, 13 * l, 156, -22 * l]
        m_matrix[3] = [-13 * l, -3 * l ** 2, -22 * l, 4 * l ** 2]
        m[i] = self.rho * self.area * l * m_matrix / 420

        # Global Stiffness Matrix
        ss[np.ix_(index, index)] += k[i]
        si[np.ix_(index, index)] += m[i]

#Barra Horizontal
start_node1 = [0, 0]
end_node1 = [1, 0]
nodes1, bars1 = create_beam(start_node1, end_node1, delta_x)
beam_1 = Beam(E, I, rho, 10, nodes1, bars1)

#Barra Vertical
start_node2 = [1, 0]
end_node2 = [1, 1]
nodes2, bars2 = create_beam(start_node2, end_node2, delta_x)
beam_2 = Beam(E, I, rho, 20, nodes2, bars2)

point_load_1 = beam_1.point_load
point_load_1[3, 0] = 0

support_1 = beam_1.support
support_1[0, :] = 0
support_1[-1, 0] = 0

# constructing the global matrix
nn = 0
beam_list=[]
for obj_name, obj in globals().items():
    if isinstance(obj, Beam):
        nn += len(obj.node)
        beam_list.append(obj)

n_dof = 2 * nn
ss = np.zeros([n_dof, n_dof])
si = np.zeros([n_dof, n_dof])

for beam in beam_list:
    analysis(beam, ss, si)

print(si, ss)
