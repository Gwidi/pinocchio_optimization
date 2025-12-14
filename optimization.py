# Easy NLP modeling in CasADi with Opti https://web.casadi.org/blog/opti/

import casadi as ca
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def  task1():
    opti = ca.Opti()

    x0 = opti.variable()
    x1 = opti.variable()
    opti.minimize(x0**2 + x1**2)
    
    opti.solver('ipopt')
    sol = opti.solve()
    

    plt.plot(sol.value(x0),sol.value(x1),'o')

    plt.show()

def  task2():
    opti = ca.Opti()

    x0 = opti.variable()
    x1 = opti.variable()
    opti.minimize((x0**2 + x1**2)**0.5)
    
    opti.solver('ipopt')
    sol = opti.solve()
    

    plt.plot(sol.value(x0),sol.value(x1),'o')

    plt.show()

def task3_4_5_6_7():
    opti = ca.Opti()
    x0 = opti.variable()
    x1 = opti.variable()
    opti.minimize(x0**2 + x1**2)
    opti.subject_to(x0 + x1 == 1)
    #opti.subject_to(2*x0 + x1 >=1)
    #opti.subject_to(x0**2 + x1**2 ==9)
    opti.subject_to(x0**2 + x1**2 >= 9)
    opti.solver('ipopt')
    sol = opti.solve()
    

    plt.plot(sol.value(x0),sol.value(x1),'o')

    plt.show()


if __name__ == '__main__':
    #task1()
    task3_4_5_6_7()