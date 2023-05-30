import numpy as np


class Fusca():
    def __init__(self) -> None:
        self.vel_vento = 100  # km/h
        self.h_carro = 0.15
        self.L_carro = 3
        self.d_dominio = 0.5 * self.L_carro
        self.H_dominio = 2 * self.L_carro

    def Y(self, x):
        return np.sqrt(((self.L_carro/2)**2) - (x - self.d_dominio - (self.L_carro/2))**2) + self.h_carro


class Phi():
    def dif_central():
        pass
