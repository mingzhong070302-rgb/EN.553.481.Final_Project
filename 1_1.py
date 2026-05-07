import numpy as np
import matplotlib.pyplot as plt

# ODE: u' = f(t,u)
def f(t, u):
    return -5*u + 5*np.cos(t) - np.sin(t)

# Exact solution
def u_exact(t):
    return np.cos(t) - np.exp(-5*t)

# Forward Euler method
def forward_euler(h, T=5):
    N = int(T / h)
    t = np.linspace(0, T, N + 1)
    u = np.zeros(N + 1)

    u[0] = 0

    for n in range(N):
        u[n+1] = u[n] + h * f(t[n], u[n])

    return t, u

# Classical RK4 method
def rk4(h, T=5):
    N = int(T / h)
    t = np.linspace(0, T, N + 1)
    u = np.zeros(N + 1)

    u[0] = 0

    for n in range(N):
        tn = t[n]
        un = u[n]

        k1 = f(tn, un)
        k2 = f(tn + h/2, un + h*k1/2)
        k3 = f(tn + h/2, un + h*k2/2)
        k4 = f(tn + h, un + h*k3)

        u[n+1] = un + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t, u

# Error computation
def global_error(t, u_num):
    return np.max(np.abs(u_num - u_exact(t)))

# Part (a): Forward Euler plot for h = 0.01
h = 0.01
t_fe, u_fe = forward_euler(h)
u_ex_fe = u_exact(t_fe)

plt.figure(figsize=(8, 5))
plt.plot(t_fe, u_ex_fe, label="Exact solution")
plt.plot(t_fe, u_fe, "--", label="Forward Euler, h=0.01")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Forward Euler vs Exact Solution")
plt.legend()
plt.grid(True)
plt.show()

# Part (b): RK4 plot for h = 0.01
t_rk, u_rk = rk4(h)
u_ex_rk = u_exact(t_rk)

plt.figure(figsize=(8, 5))
plt.plot(t_rk, u_ex_rk, label="Exact solution")
plt.plot(t_rk, u_rk, "--", label="RK4, h=0.01")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("RK4 vs Exact Solution")
plt.legend()
plt.grid(True)
plt.show()

# Part (c): convergence table
hs = [0.01, 0.005, 0.001]

fe_errors = []
rk4_errors = []

for h in hs:
    t_fe, u_fe = forward_euler(h)
    t_rk, u_rk = rk4(h)

    fe_errors.append(global_error(t_fe, u_fe))
    rk4_errors.append(global_error(t_rk, u_rk))

# Compute observed orders
fe_orders = [None]
rk4_orders = [None]

for i in range(1, len(hs)):
    p_fe = np.log(fe_errors[i-1] / fe_errors[i]) / np.log(hs[i-1] / hs[i])
    p_rk = np.log(rk4_errors[i-1] / rk4_errors[i]) / np.log(hs[i-1] / hs[i])

    fe_orders.append(p_fe)
    rk4_orders.append(p_rk)

# Print table nicely
print("\nForward Euler Convergence Table")
print("-" * 65)
print(f"{'Step size h':<15}{'Global error':<25}{'Observed order':<20}")
print("-" * 65)

for h, err, p in zip(hs, fe_errors, fe_orders):
    if p is None:
        print(f"{h:<15.5f}{err:<25.10e}{'-':<20}")
    else:
        print(f"{h:<15.5f}{err:<25.10e}{p:<20.4f}")

print("\nRK4 Convergence Table")
print("-" * 65)
print(f"{'Step size h':<15}{'Global error':<25}{'Observed order':<20}")
print("-" * 65)

for h, err, p in zip(hs, rk4_errors, rk4_orders):
    if p is None:
        print(f"{h:<15.5f}{err:<25.10e}{'-':<20}")
    else:
        print(f"{h:<15.5f}{err:<25.10e}{p:<20.4f}")