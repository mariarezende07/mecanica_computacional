import matplotlib.pyplot as plt
import numpy as np


class Fusca():
    def __init__(self) -> None:
        self.V = 100/3.6  # m/s
        self.h_carro = 0.15
        self.L_carro = 3
        self.d_dominio = 0.5 * self.L_carro
        self.H_dominio = 2 * self.L_carro

        self.x_dominio = 2 * self.d_dominio + self.L_carro
        self.y_dominio = self.H_dominio

        self.Nx = 100
        self.Ny = 100

        # Create the grid
        self.x = np.linspace(0, self.x_dominio, self.Nx)
        self.y = np.linspace(0, self.y_dominio, self.Ny)

        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.A = np.zeros((self.Nx*self.Ny, self.Nx*self.Ny))
        self.b = np.zeros(self.Nx*self.Ny)

    def car_height(self, x):
        return np.sqrt(((self.L_carro/2)**2) - (x - self.d_dominio - (self.L_carro/2))**2) + self.h_carro

    def index(self, i, j):
        return j * self.Nx + i

    def mdf(self):
        for i in range(self.Nx):
            for j in range(self.Ny):
                k = self.index(i, j)
                if j == 0:
                    self.A[k, k] = 1
                    self.b[k] = 0
                else:
                    x_coord = self.x[i]
                    y_coord = self.y[j]
                    if i == 0 or i == self.Nx - 1:
                        self.A[k, k] = -1.0
                        self.A[k, self.index(i - 1, j)] = 1.0
                        self.b[k] = 0.0
                    elif j == self.Ny - 1:
                        self.A[k, k] = -3.0
                        self.A[k, self.index(i, j - 1)] = 4.0
                        self.b[k] = 4.0 * self.V * \
                            (1.0 / (self.y_dominio/self.Ny))
                    elif (0 <= y_coord <= self.car_height(x_coord)) and (self.d_dominio <= x_coord <= self.d_dominio + self.L_carro):
                        self.A[k, k] = 1
                        self.b[k] = 0
                    else:
                        self.A[k, k] = -4.0
                        self.A[k, self.index(i - 1, j)] = 1.0
                        self.A[k, self.index(i + 1, j)] = 1.0
                        self.A[k, self.index(i, j - 1)] = 1.0
                        self.A[k, self.index(i, j + 1)] = 1.0

        T = np.linalg.solve(self.A, self.b)
        T = T.reshape((self.Ny, self.Nx))

        return T
    

    def plot(self):

        T = self.mdf()
        print(T)
        plt.contour(self.X, self.Y, T, levels=20)
        plt.colorbar(label='Temperature')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Temperature Distribution')
        plt.show()


fusca = Fusca()
fusca.plot()
