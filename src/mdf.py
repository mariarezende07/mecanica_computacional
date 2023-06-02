import matplotlib.pyplot as plt
import numpy as np


class Fusca():
    def __init__(self) -> None:
        self.V = 100/3.6  # m/s
        self.h_carro = 0.15
        self.L_carro = 3
        self.d_dominio = 1.5

        self.x_dominio = (2*self.d_dominio + self.L_carro)
        self.y_dominio = 6

        self.Nx = 200
        self.Ny = 200
        self.psi = np.transpose(self.setup_matrix())

    def car_height(self, x):
        return np.sqrt(((self.L_carro/2)**2) - (x - self.d_dominio - (self.L_carro/2))**2) + self.h_carro

    def inside_car(self, x, y):
        return (self.distance_from_circle(x, y) < self.L_carro/2) & (y > self.h_carro)

    def circle_bottom_border(self, y_pos, x_pos, delta):

        return (
            (self.h_carro - y_pos < delta)
            &
            (y_pos < self.h_carro)
            &
            (x_pos > self.d_dominio)
            &
            (x_pos < self.d_dominio + self.L_carro)
        )

    def distance_from_circle(self, x_pos, y_pos):
        return np.sqrt((x_pos - self.d_dominio - self.L_carro/2) ** 2 + (y_pos - self.h_carro)**2)

    def setup_matrix(self):
        # Define zero matrix with 2D linearization to 1D space
        psi = np.zeros((self.Nx, self.Ny))

        x = np.linspace(0, self.x_dominio, self.Nx)
        y = np.linspace(0, self.y_dominio, self.Ny)

        max_iter = 1000
        tolerance = 1e-2
        omega = 1.85
        for _ in range(max_iter):
            psi_old = psi.copy()
            for i in range(self.Nx):
                for j in range(self.Ny):
                    psi_old[i, j] = psi[i, j].copy()
                    pos_x = x[i]
                    pos_y = y[j]

                    delta = 1/self.Ny

                    if j == 0:
                        continue
                    elif j == self.Ny - 1:
                        if i == 0:
                            psi[i, j] = (delta * self.V +
                                         psi[i+1, j] + psi[i, j-1])/2
                        elif i == self.Nx - 1:
                            psi[i, j] = (delta * self.V +
                                         psi[i-1, j] + psi[i, j-1])/2
                        else:
                            psi[i, j] = (
                                2 * (psi[i, j-1] + delta * self.V) + psi[i+1, j] + psi[i-1, j])/4
                    elif i == 0:
                        psi[i, j] = (2*psi[i+1, j] +
                                     psi[i, j-1] + psi[i, j+1])/4
                    elif i == self.Nx - 1:
                        psi[i, j] = (2*psi[i-1, j] +
                                     psi[i, j-1] + psi[i, j+1])/4

                    elif self.inside_car(pos_x, pos_y):
                        continue
                    # Circle bottom border
                    elif self.circle_bottom_border(x_pos=pos_x, y_pos=pos_y, delta=delta):
                        a = ((self.h_carro - pos_y) / delta)
                        psi[i, j] = ((psi[i+1, j]+psi[i-1, j] +
                                     (2*psi[i, j - 1]) / (a+1))/(2/a + 2))

                    elif (self.distance_from_circle(pos_x, pos_y) - self.L_carro/2 < delta) and (self.distance_from_circle(pos_x, pos_y) > self.L_carro/2) and (pos_y > self.h_carro):
                        if pos_x < self.x_dominio/2:
                            # Left circle border
                            continue
                        else:
                            # Right circle border
                            continue

                        
                    else:
                        psi[i, j] = (
                            (psi[i + 1, j] + psi[i - 1, j] + psi[i, j + 1] + psi[i, j - 1]) / 4)

                    psi[i, j] = (1 - omega) * psi_old[i, j] + \
                        omega * psi[i, j]
            print(np.nanmax(np.abs((psi - psi_old)/psi)))
            if np.nanmax(np.abs((psi - psi_old)/psi)) < tolerance:
                break

        return psi

    def plot(self):

        # Define meshgrid
        x = np.linspace(0, self.x_dominio, self.Nx)
        y = np.linspace(0, self.y_dominio, self.Ny)
        X, Y = np.meshgrid(x, y)

        # Init Setup matrix

        psi = self.psi

        # Plot contour
        plt.contour(X, Y, psi, levels=20)
        plt.colorbar(label='Phi')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Phi Distribution')
        plt.show()

    def partial_velocities(self):
        psi = self.psi
        partial_x, partial_y = np.gradient(psi)
        x = np.arange(psi.shape[1])
        y = np.arange(psi.shape[0])


fusca = Fusca()
fusca.plot()
