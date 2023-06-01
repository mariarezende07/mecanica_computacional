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
        psi = np.zeros((self.Nx, self.Ny))

        x = np.linspace(0, self.x_dominio, self.Nx)
        y = np.linspace(0, self.y_dominio, self.Ny)

        max_iter = 1000
        tolerance = 1e-2
        omega = 1.85
        for iteration in range(max_iter):
            psi_old = psi.copy()
            for i in range(self.Nx):
                for j in range(self.Ny):
                    delta_x = x[i]
                    delta_y = y[j]
                    if i == 0:
                        psi[i, j] = psi[i+1, j]
                    elif i == self.Nx - 1:
                        psi[i-1, j] = psi[i-2, j]
                    elif j == 0:
                        continue
                    elif j == self.Ny - 1:
                        psi[i, j-1] = psi[i, j-2] + self.V * delta_y
                    elif (0 <= delta_y <= self.car_height(delta_x)) and (self.d_dominio <= delta_x <= self.d_dominio + self.L_carro):
                        continue
                    else:
                        psi[i, j] = (1 - omega) * psi_old[i, j] + omega * ((psi[i + 1, j] + psi[i - 1, j] +
                                                                           psi[i, j + 1] + psi[i, j - 1]) / 4)
            print(psi)

            if np.max(np.abs(psi - psi_old)) < tolerance:
                break

        return psi

    def plot(self):

        # Define meshgrid
        x = np.linspace(0, self.x_dominio, self.Nx)
        y = np.linspace(0, self.y_dominio, self.Ny)
        X, Y = np.meshgrid(x, y)

        # Init Setup matrix

        psi = self.setup_matrix()
        print(psi)

        # Plot contour
        plt.contour(X, Y, psi, levels=20)
        plt.colorbar(label='Phi')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Phi Distribution')
        plt.show()


fusca = Fusca()
fusca.plot()
