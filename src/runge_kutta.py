import numpy as np


def rk4(f, t0, tf, x0, n):
    """ Resolve uma equação diferencial pelo metodo de Runge Kutta de quarta ordem:
        f (function):  retorna a função que descreve as equações e o vetor tempo  
        t0 (int): inicio do intervalo de tempo
        tf (int): fim do intervalo de tempo
        x0 (array): Condições iniciais
        n (int): Numero de interadas
    """
    h = (tf - t0) / n
    t = np.linspace(t0, tf, n+1)
    x = np.zeros((n+1, len(x0)))
    x[0] = x0
    for i in range(n):
        k1 = f(t[i], x[i])
        k2 = f(t[i] + h/2, x[i] + h*k1/2)
        k3 = f(t[i] + h/2, x[i] + h*k2/2)
        k4 = f(t[i] + h, x[i] + h*k3)
        x[i+1] = x[i] + h*(k1 + 2*k2 + 2*k3 + k4) / 6
    return t, x
