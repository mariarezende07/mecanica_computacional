import matplotlib.pyplot as plt
import numpy as np


class Fusca():
    def __init__(self) -> None:
        self.V = 27.78  # m/s
        self.h_carro = 0.15
        self.L_carro = 3
        self.d_dominio = 1.5

        self.x_dominio = 6
        self.y_dominio = 6

        self.Nx = 100
        self.Ny = 100

    def car_height(self, x):
        return np.sqrt(((self.L_carro/2)**2) - (x - self.d_dominio - (self.L_carro/2))**2) + self.h_carro

    def index(self, i, j):
        return j * self.Nx + i

    def setup_matrix(self):
        # Define zero matrix with 2D linearization to 1D space
        A = np.zeros((self.Nx*self.Ny, self.Nx*self.Ny))
        b = np.zeros(self.Nx*self.Ny)

        x = np.linspace(0, self.x_dominio, self.Nx)
        y = np.linspace(0, self.y_dominio, self.Ny)

        for i in range(self.Nx):
            for j in range(self.Ny):
                k = self.index(i, j)
                if j == 0:
                    A[k, k] = 1
                    b[k] = 0
                else:
                    x_coord = x[i]
                    y_coord = y[j]
                    if i == 0 or i == self.Nx - 1:
                        A[k, k] = -1.0
                        A[k, self.index(i - 1, j)] = 1.0
                        b[k] = 0.0
                    elif j == self.Ny - 1:
                        A[k, k] = -3.0
                        A[k, self.index(i, j - 1)] = 4.0
                        b[k] = 4.0 * self.V * \
                            (1.0 / (self.y_dominio/self.Ny))
                    elif (0 <= y_coord <= self.car_height(x_coord)) and (self.d_dominio <= x_coord <= self.d_dominio + self.L_carro):
                        A[k, k] = 1
                        b[k] = 0
                    else:
                        A[k, k] = -4.0
                        A[k, self.index(i - 1, j)] = 1.0
                        A[k, self.index(i + 1, j)] = 1.0
                        A[k, self.index(i, j - 1)] = 1.0
                        A[k, self.index(i, j + 1)] = 1.0

        return A, b

    def plot(self):

        # Define meshgrid
        x = np.linspace(0, self.x_dominio, self.Nx)
        y = np.linspace(0, self.y_dominio, self.Ny)
        X, Y = np.meshgrid(x, y)

        # Init Setup matrix

        A, b = self.setup_matrix()
        T = np.linalg.solve(A, b)
        T = T.reshape((self.Ny, self.Nx))
        print(T)

        # Plot contour
        plt.contour(X, Y, T, levels=20)
        plt.colorbar(label='Phi')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Phi Distribution')
        plt.show()


fusca = Fusca()
fusca.plot()
