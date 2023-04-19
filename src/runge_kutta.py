import numpy as np

def f(t, x):
    """Local para escrever as equações diferenciais e suas derivadas:
        t (np.array):  Vetor para armazenar os deltas t   
        x (np.array): Vetor para armazenar os valores das n interações:
        float: Volume do cubo
    """
    x1, x2, x3, x4 = x
    dx1 = x2
    dx2 = 2*x3**2 - x2 + 3*x4
    dx3 = x4
    dx4 = x1 + x3*np.sin(x1*t/2) - 3*x4
    print(np.array([dx1, dx2, dx3, dx4]))
    return np.array([dx1, dx2, dx3, dx4])

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

# caso de uso: Deu certo para esse caso, bateu com oq eu calculei em aula 
t0 = 0
tf = 0.2
x0 = np.array([1, 3, 2, 1/3])
n = 1
t, x = rk4(f, t0, tf, x0, n)
