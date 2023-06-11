import matplotlib.pyplot as plt
import numpy as np


class Fusca():
    def __init__(self, use_saved_phi=True, use_saved_T=True) -> None:
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
        if use_saved_phi:
            with open('src/utils/psi_matrix.npy', 'rb') as f:
                self.psi = np.load(f)
        else:
            with open('src/utils/psi_matrix.npy', 'wb') as f:
                self.psi = np.transpose(self.calculate_phi())
                np.save(f, self.psi)

        self.p_atm = 101.325
        self.rho = 1.25
        self.gamma_ar = 1.4
        self.cp = 1002
        self.k = 0.026
        self.T_fora = 20
        self.T_motor = 80
        self.T_dentro = 25

        if use_saved_T:
            with open('src/utils/T_matrix.npy', 'rb') as f:
                self.Temperature = np.load(f)
        else:
            with open('src/utils/T_matrix.npy', 'wb') as f:
                self.Temperature = np.transpose(self.calc_temperature())
                np.save(f, self.Temperature)

    def car_height(self, x):
        return np.sqrt(((self.L_carro/2)**2) - (x - self.d_dominio - (self.L_carro/2))**2) + self.h_carro

    def inside_car(self, x, y):
        return (self.distance_from_circle(x, y) < self.L_carro/2) and (y > self.h_carro)

    def inside_car_motor(self, x, y):
        current_position = np.array([y, x])
        center = np.array([self.h_carro, self.L_carro/2 + self.d_dominio])
        end_of_car = np.array([self.h_carro, self.L_carro + self.d_dominio])

        ba = current_position - center
        bc = end_of_car - center

        cosine_angle = np.dot(ba, bc) / \
            (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle < 60

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

    def calculate_phi(self):
        # Define zero matrix with 2D linearization to 1D space
        psi = np.zeros((self.Nx, self.Ny))

        delta = self.delta

        x = self.x
        y = self.y
        max_iter = 10000
        tolerance = 1e-4
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

    def pressure_calc_in_car(self):
        p = np.zeros((self.Nx, self.Ny))
        u, v = self.calc_partial_velocities()
        delta = self.delta
        for i in range(self.Nx):
            for j in range(self.Ny):
                pos_x = self.x[i]  # Current position at x axis
                pos_y = self.y[j]  # Current position at y axis
                if ((self.distance_from_circle(pos_x, pos_y) - self.L_carro/2 < delta) and (self.distance_from_circle(pos_x, pos_y) > self.L_carro/2) and (pos_y > self.h_carro)) or self.circle_bottom_border(x_pos=pos_x, y_pos=pos_y, delta=delta):
                    first_term = self.p_atm + self.rho * \
                        ((self.gamma_ar - 1)/self.gamma_ar)
                    second_term = ((self.V**2)/2) - \
                        ((np.sqrt(u[i, j]**2 + v[i, j]**2))**2)/2
                    p[i, j] = first_term * second_term

        return np.transpose(p)

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
        u, v = np.gradient(np.transpose(self.psi), self.delta)
        return u, -v

    def plot_partial_velocities(self):
        x = np.arange(0, self.x_dominio, self.delta)
        y = np.arange(0, self.y_dominio, self.delta)
        X, Y = np.meshgrid(x, y)

        u, v = self.calc_partial_velocities()
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(u, cmap='plasma', origin='lower', extent=[
                   0, self.Nx * self.delta, 0, self.Ny * self.delta])
        plt.colorbar(label='Gradient along x-direction')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Gradient along x-direction')

        plt.subplot(1, 2, 2)
        plt.imshow(v, cmap='plasma', origin='lower', extent=[
                   0, self.Nx * self.delta, 0, self.Nx * self.delta])
        plt.colorbar(label='Gradient along y-direction')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Gradient along y-direction')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.quiver(X[::3, ::3], Y[::3, ::3],
                   u[::3, ::3], v[::3, ::3], pivot='mid')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Velocities Field')
        plt.grid(True)
        plt.show()

    def pressure_calc_in_domain(self):
        p = np.zeros((self.Nx, self.Ny))
        u, v = self.calc_partial_velocities()

        for i in range(self.Nx):
            for j in range(self.Ny):
                first_term = self.p_atm + self.rho * \
                    ((self.gamma_ar - 1)/self.gamma_ar)
                second_term = ((self.V**2)/2) - \
                    ((np.sqrt(u[i, j]**2 + v[i, j]**2))**2)/2
                p[i, j] = first_term * second_term

        return p

    def plot_pressure_heatmap(self):
        p = self.pressure_calc_in_domain()
        print(p)
        # Define meshgrid
        x = np.arange(0, self.x_dominio, self.delta)
        y = np.arange(0, self.y_dominio, self.delta)
        X, Y = np.meshgrid(x, y)

        # Plot contour
        plt.figure(figsize=(8, 6))
        plt.imshow(p, cmap='plasma', origin='lower', extent=[
                   0, self.Nx * self.delta, 0, self.Nx * self.delta])
        plt.colorbar(label='Pressure')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Pressure Distribution')
        plt.show()

    def plot_pressure_heatmap_in_car(self):
        p = self.pressure_calc_in_car()
        min_index = np.unravel_index(np.argmin(p), p.shape)
        x = np.arange(0, self.x_dominio, self.delta)
        y = np.arange(0, self.y_dominio, self.delta)
        # Plot contour
        plt.figure(figsize=(8, 6))
        plt.imshow(p, cmap='Purples', origin='lower', extent=[
                   0, self.Nx * self.delta, 0, self.Nx * self.delta])
        plt.colorbar(label='Pressure')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Pressure Distribution')
        print(min_index)
        plt.scatter(x[min_index[1]], y[min_index[0]], s=12**2)

        plt.show()

    def calc_lift_force(self):
        p = self.pressure_calc_in_car()
        x = np.arange(0, self.x_dominio, self.delta)
        y = np.arange(0, self.y_dominio, self.delta)

        Fx = np.sum(p * x * self.delta)
        Fy = np.sum(p * y * self.delta)

        return Fx, Fy

    def calc_temperature(self):
        u, v = self.calc_partial_velocities()

        T = np.zeros((self.Nx, self.Ny))
        x = self.x
        y = self.y

        max_iter = 1
        tolerance = 1e-4
        omega = 1.15
        for _ in range(max_iter):
            T_old = T.copy()
            for i in range(self.Nx):
                for j in range(self.Ny):
                    T_old[i, j] = T[i, j].copy()
                    pos_x = x[i]  # Current position at x axis
                    pos_y = y[j]  # Current position at y axis

                    alpha = ((self.rho * self.cp)/(self.k * 2)) * self.delta

                    if j == 0:  # Bottom border
                        if i == 0:  # Bottom left border
                            T[i, j] = self.T_fora

                        elif i == self.Nx - 1:  # Bottom right border
                            T[i, j] = (2*T[i-1, j] + T[i, j+1] + T[i, j-1])/3
                        else:  # Bottom inner border
                            if u[i, j] > 0:
                                T[i, j] = ((2*T[i, j-1] + T[i+1, j] + T[i-1, j])/4 + (alpha/4)
                                           * u[i, j] * T[i-1, j]) / (1 + (alpha/4) * u[i, j])
                            else:  # u < 0
                                T[i, j] = ((2*T[i, j-1] + T[i+1, j] + T[i-1, j])/4 + (alpha/4)
                                           * u[i, j] * T[i+1, j]) / (1 - (alpha/4) * u[i, j])
                    elif j == self.Ny - 1:  # Top border
                        if i == 0:  # Top left border
                            T[i, j] = self.T_fora
                        elif i == self.Nx - 1:  # Top right border
                            T[i, j] = (T[i-1, j] +
                                       T[i, j-1])/2
                        else:  # Top inner border
                            if u[i, j] > 0:
                                T[i, j] = ((2*T[i, j-1] + T[i+1, j] + T[i-1, j])/4 + (alpha/4)
                                           * u[i, j] * T[i-1, j]) / (1 + (alpha/4) * u[i, j])
                            else:  # u < 0
                                T[i, j] = ((2*T[i, j-1] + T[i+1, j] + T[i-1, j])/4 + (alpha/4)
                                           * u[i, j] * T[i+1, j]) / (1 - (alpha/4) * u[i, j])
                    elif i == 0:  # Left inner border
                        T[i, j] = self.T_fora
                    elif i == self.Nx - 1:  # Right inner border
                        if v[i, j] > 0:
                            T[i, j] = ((alpha * v[i, j] * T[i, j-1]) + 2 *
                                       T[i-1, j] + 2 * T[i, j-1])/(4 + alpha * v[i, j])
                        else:  # v < 0
                            T[i, j] = ((alpha * v[i, j] * T[i, j+1]) + 2 *
                                       T[i-1, j] + 2 * T[i, j-1])/(4 - alpha * v[i, j])

                    elif self.inside_car(pos_x, pos_y):
                        
                        T[i, j] = self.T_motor if self.inside_car_motor(
                            pos_x, pos_y) else self.T_dentro

                    else:  # Inner points
                        laplace_term = (
                            T[i+1, j] + T[i-1, j]+T[i, j+1] + T[i, j-1])/4
                        if u[i, j] > 0 and v[i, j] > 0:

                            partial_term = (alpha / 4) * \
                                ((u[i, j] * T[i-1, j] + v[i, j] * T[i, j-1]))
                            divisive_term = - (
                                1 + (alpha/4) * (u[i, j] + v[i, j]))
                        elif u[i, j] < 0 and v[i, j] < 0:
                            partial_term = (
                                alpha / 4) * ((u[i, j] * T[i+1, j] + v[i, j] * T[i, j+1]))
                            divisive_term = (
                                1 + (alpha/4) * (u[i, j] + v[i, j]))
                        elif u[i, j] < 0 and v[i, j] > 0:
                            partial_term = (
                                alpha / 4) * ((u[i, j] * T[i+1, j] - v[i, j] * T[i, j-1]))
                            divisive_term = (
                                1 + (alpha/4) * (u[i, j] - v[i, j]))
                        else:  # u > 0 and v < 0
                            partial_term = (
                                alpha / 4) * ((- u[i, j] * T[i-1, j] + v[i, j] * T[i, j+1]))
                            divisive_term = (
                                1 + (alpha/4) * (-u[i, j] + v[i, j]))

                        T[i, j] = (laplace_term + partial_term) / \
                            divisive_term

                    T[i, j] = (1 - omega) * T[i, j] + \
                        omega * T[i, j]
            print(np.nanmax(np.abs((T - T_old)/T)))
            if np.nanmax(np.abs((T - T_old)/T)) < tolerance:
                break

        return T

    def plot_temperature(self):

        # Define meshgrid
        x = np.arange(0, self.x_dominio, self.delta)
        y = np.arange(0, self.y_dominio, self.delta)
        X, Y = np.meshgrid(x, y)

        # Plot contour
        print(self.Temperature)
        plt.imshow(self.Temperature, cmap='hot_r', origin='lower', extent=[
                   0, self.Nx * self.delta, 0, self.Ny * self.delta])
        plt.colorbar(label='Temperature')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Temperature Distribution')
        plt.show()


fusca = Fusca(use_saved_phi=True, use_saved_T=False)
print(fusca.plot_pressure_heatmap_in_car())
