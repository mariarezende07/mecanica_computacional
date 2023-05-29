import pandas as pd
import numpy as np


class Carro():
    def __init__(self, velocidade=50, amortecimentos=3e4, molas=2.8e7):
        # Contanstes iniciais
        self.v = velocidade
        self.k1 = molas
        self.k2 = molas
        self.c1 = amortecimentos
        self.c2 = amortecimentos
        self.M = 1783
        self.a = 1220e-03
        self.b = 1500e-03
        self.Ic = 4000
        self.e = 0.75
        self.L = 0.5
        self.A = 60e-03
        self.f = 0.35
        self.w = (2 * np.pi() * velocidade)/self.L
        self.me = 20
        self.fe = 2100
        self.r = 0.045

        self.we = 2 * np.pi() * self.fe
        self.Fn = self.me * self.we ** 2 * self.r

    def d1(self, t):
        if t < 2:
            return self.A * (1 - np.cos(self.w * t))
        return 0

    def d2(self, t):
        if t < 2:
            return self.A * (1 + np.cos(self.w * t))
        return 0

    def d1_dot(self, t):
        if t < 2:
            return self.A * self.w * np.sin(self.w * t)
        return 0

    def d2_dot(self, t):
        if t < 2:
            return - self.A * self.w * np.sin(self.w * t)
        return 0

    def car_function(self, x, t):
        x1, x2, x3, x4 = x
        # x1 = x
        # x2 = x_dot
        # x3 = theta
        # x4 = theta_dot
        dx1 = x2
        dx2 = (-self.k1 * (x1 - self.a * x3 - self.d1(t)) -
               self.k2*(x1 + self.b * x3 - self.d2(t)) -
               self.c1 * (x2 - self.a * x4 - self.d1_dot(t)) -
               self.c2 * (x2 + self.b * x4 - self.d2_dot) +
               self.Fn * np.sin(self.we * t))/self.M
        dx3 = x4
        dx4 = (self.k1*(x - self.a*x3 - self.d1(t))*self.a
               - self.k2*(x + self.b*x3 - self.d2(t))*self.b
               + self.c1*(x2 - self.a*x4 - self.dd1(t)) *
               self.a - self.c2*(x2 + self.b*x4 - self.dd2(t))*self.b - self.Fn *
               np.sin(self.we*t)*self.e - self.Fn *
               np.cos(self.omega_e*t)*self.f)/self.Ic
        
        print(np.array([dx1, dx2, dx3, dx4]))
        return np.array([dx1, dx2, dx3, dx4])

    