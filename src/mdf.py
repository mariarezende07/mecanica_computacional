import matplotlib.pyplot as plt
import numpy as np


class Fusca():
    def __init__(self, use_saved_matrix) -> None:
        self.V = 100/3.6  # m/s
        self.h_carro = 0.15
        self.L_carro = 3
        self.d_dominio = 1.5

        self.x_dominio = (2*self.d_dominio + self.L_carro)
        self.y_dominio = 6

        self.delta = 0.05
        self.x = np.arange(0, self.x_dominio, self.delta)
        self.y = np.arange(0, self.y_dominio, self.delta)

        self.Nx = len(self.x)
        self.Ny = len(self.y)
        if use_saved_matrix:
            with open('utils/psi_matrix.npy', 'rb') as f:
                self.psi = np.load(f)
        else:
            with open('utils/psi_matrix.npy', 'wb') as f:
                self.psi = np.transpose(f, self.setup_matrix())
                np.save(self.psi)

    def car_height(self, x):
        return np.sqrt(((self.L_carro/2)**2) - (x - self.d_dominio - (self.L_carro/2))**2) + self.h_carro

    def inside_car(self, x, y):
        return (self.distance_from_circle(x, y) < self.L_carro/2) and (y > self.h_carro)

    def circle_bottom_border(self, y_pos, x_pos, delta):

        return (
            (self.h_carro - y_pos < delta)
            and
            (y_pos < self.h_carro)
            and
            (x_pos > self.d_dominio)
            and
            (x_pos < self.d_dominio + self.L_carro)
        )

    def right_circle_border(self, y_pos, x_pos, delta):
        a = (
            x_pos
            -
            self.d_dominio
            -
            self.L_carro/2
            -
            np.sqrt(
                (self.L_carro/2)**2
                -
                (y_pos - self.h_carro)**2
            )
        ) / delta

        b = (
            y_pos
            -
            self.h_carro
            -
            np.sqrt(
                (self.L_carro/2)**2
                -
                (self.d_dominio + self.L_carro/2 - x_pos)**2
            )
        ) / delta

        return a, b

    def left_circle_border(self, y_pos, x_pos, delta):
        a = (
            self.d_dominio
            +
            self.L_carro/2
            -
            x_pos
            -
            np.sqrt(
                (self.L_carro/2)**2
                -
                (y_pos - self.h_carro)**2
            )
        ) / delta

        b = (
            y_pos
            -
            self.h_carro
            -
            np.sqrt(
                (self.L_carro/2)**2
                -
                (self.d_dominio + self.L_carro/2 - x_pos)**2
            )
        ) / delta

        return a, b

    def distance_from_circle(self, x_pos, y_pos):
        return np.sqrt((x_pos - self.d_dominio - self.L_carro/2) ** 2 + (y_pos - self.h_carro)**2)

    def setup_matrix(self):
        # Define zero matrix with 2D linearization to 1D space
        psi = np.zeros((self.Nx, self.Ny))

        delta = self.delta

        x = self.x
        y = self.y
        max_iter = 1000
        tolerance = 1e-2
        omega = 1.85
        for _ in range(max_iter):
            psi_old = psi.copy()
            for i in range(self.Nx):
                for j in range(self.Ny):
                    psi_old[i, j] = psi[i, j].copy()
                    pos_x = x[i]  # Current position at x axis
                    pos_y = y[j]  # Current position at y axis

                    if j == 0:  # Bottom border points
                        continue
                    elif j == self.Ny - 1:  # Top border points
                        if i == 0:  # Top left corner points
                            psi[i, j] = (delta * self.V +
                                         psi[i+1, j] + psi[i, j-1])/2
                        elif i == self.Nx - 1:  # Top right corner points
                            psi[i, j] = (delta * self.V +
                                         psi[i-1, j] + psi[i, j-1])/2
                        else:  # Inner top points
                            psi[i, j] = (
                                2 * (psi[i, j-1] + delta * self.V) + psi[i+1, j] + psi[i-1, j])/4
                    elif i == 0:  # Left border points
                        psi[i, j] = (2*psi[i+1, j] +
                                     psi[i, j-1] + psi[i, j+1])/4
                    elif i == self.Nx - 1:  # RIght border points
                        psi[i, j] = (2*psi[i-1, j] +
                                     psi[i, j-1] + psi[i, j+1])/4

                    # Circle bottom border
                    elif self.circle_bottom_border(x_pos=pos_x, y_pos=pos_y, delta=delta):
                        a = ((self.h_carro - pos_y) / delta)
                        psi[i, j] = ((psi[i+1, j]+psi[i-1, j] +
                                     (2*psi[i, j - 1]) / (a+1))/(2/a + 2))

                    # Check if point is in the border
                    elif (self.distance_from_circle(pos_x, pos_y) - self.L_carro/2 < delta) and (self.distance_from_circle(pos_x, pos_y) > self.L_carro/2) and (pos_y > self.h_carro):

                        if pos_x < self.x_dominio/2:
                            # Left circle border
                            a, b = self.left_circle_border(
                                x_pos=pos_x, y_pos=pos_y, delta=delta)
                            if (a < 1 and b < 1):
                                psi[i, j] = (
                                    2*a*psi[i-1, j]
                                    /
                                    (a*(1+a))
                                    +
                                    2*(b*psi[i, j+1])
                                    /
                                    (b*(1+b))
                                ) / (2/a + 2/b)
                            elif (a < 1 and not b < 1):
                                a, b = self.right_circle_border(
                                    x_pos=pos_x, y_pos=pos_y, delta=delta)
                                psi[i, j] = (
                                    2*a*psi[i-1, j]
                                    /
                                    (a*(1+a))
                                    +
                                    (psi[i, j-1] + psi[i, j+1])
                                ) / (2/a + 2)
                            elif (b < 1 and not a < 1):
                                psi[i, j] = (
                                    (psi[i+1, j] + psi[i-1, j])
                                    +
                                    2*(b*psi[i, j+1])
                                    /
                                    (b*(1+b))
                                ) / (2 + 2/b)
                            else:
                                psi[i, j] = (
                                    (psi[i + 1, j] + psi[i - 1, j] + psi[i, j + 1] + psi[i, j - 1]) / 4)

                        else:
                            # Right circle border
                            if (a < 1 and b < 1):
                                psi[i, j] = (
                                    2*a*psi[i+1, j]
                                    /
                                    (a*(1+a))
                                    +
                                    2*(b*psi[i, j+1])
                                    /
                                    (b*(1+b))
                                ) / (2/a + 2/b)
                            elif (a < 1 and not b < 1):
                                psi[i, j] = (
                                    2*a*psi[i+1, j]
                                    /
                                    (a*(1+a))
                                    +
                                    (psi[i, j-1] + psi[i, j+1])
                                ) / (2/a + 2)
                            elif (b < 1 and not a < 1):
                                psi[i, j] = (
                                    (psi[i-1, j] + psi[i+1, j])
                                    +
                                    2*(b*psi[i, j+1])
                                    /
                                    (b*(1+b))
                                ) / (2 + 2/b)
                            else:
                                psi[i, j] = (
                                    (psi[i + 1, j] + psi[i - 1, j] + psi[i, j + 1] + psi[i, j - 1]) / 4)
                    # Check if point is inside car
                    elif self.inside_car(pos_x, pos_y):
                        continue

                    else:  # Inner points
                        psi[i, j] = (
                            (psi[i + 1, j] + psi[i - 1, j] + psi[i, j + 1] + psi[i, j - 1]) / 4)

                    psi[i, j] = (1 - omega) * psi_old[i, j] + \
                        omega * psi[i, j]

            if np.nanmax(np.abs((psi - psi_old)/psi)) < tolerance:
                break

        return psi

    def plot_psi(self):

        # Define meshgrid
        x = np.arange(0, self.x_dominio, self.delta)
        y = np.arange(0, self.y_dominio, self.delta)
        X, Y = np.meshgrid(x, y)

        # Plot contour
        plt.contour(X, Y,  self.psi, levels=20)
        plt.colorbar(label='Phi')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Phi Distribution')
        plt.show()

    def calc_partial_velocities(self):
        x = np.arange(0, self.x_dominio, self.delta)
        y = np.arange(0, self.y_dominio, self.delta)
        partial_x = np.gradient(self.psi, x)
        partial_y = -np.gradient(self.psi, y)
        return partial_x, partial_y

    def plot_partial_velocities(self):
        pass


fusca = Fusca()
fusca.partial_velocities()
