import numpy as np
import matplotlib.pyplot as plt

def heat_exact_solution(x, t, nu=0.01):
    return (
        np.exp(-nu * np.pi**2 * t) * np.sin(np.pi * x)
        + 0.5 * np.exp(-9 * nu * np.pi**2 * t) * np.sin(3 * np.pi * x)
    )


def heat_forward_euler(dx=1/64, r=0.4, T=0.5, nu=0.01):
    # Compute dt from r = nu dt / dx^2
    dt = r * dx**2 / nu

    # Adjust number of time steps so final time is exactly T
    Nt = int(np.ceil(T / dt))
    dt = T / Nt
    r = nu * dt / dx**2

    # Spatial grid
    Nx = int(1 / dx)
    x = np.linspace(0, 1, Nx + 1)

    # Time grid
    t = np.linspace(0, T, Nt + 1)

    # Solution array: rows are time, columns are space
    u = np.zeros((Nt + 1, Nx + 1))

    # Initial condition
    u[0, :] = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)

    # Boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0

    # Forward Euler time stepping
    for n in range(Nt):
        for i in range(1, Nx):
            u[n+1, i] = u[n, i] + r * (
                u[n, i+1] - 2*u[n, i] + u[n, i-1]
            )

    return x, t, u, dt, r


# Run finite-difference solver
x_fd, t_fd, u_fd, dt_fd, r_fd = heat_forward_euler(dx=1/64, r=0.4, T=0.5, nu=0.01)

print("=" * 60)
print("Problem 2.1: Finite-Difference Reference Solution")
print("=" * 60)
print(f"Delta x = {1/64:.10e}")
print(f"Delta t = {dt_fd:.10e}")
print(f"r = {r_fd:.10e}")

# Compute L2 error at t = 0.5
u_exact_final = heat_exact_solution(x_fd, 0.5, nu=0.01)
error_final = u_fd[-1, :] - u_exact_final

# Discrete L2 error
l2_error = np.sqrt(np.sum(error_final**2) * (x_fd[1] - x_fd[0]))

print(f"L2 error at t = 0.5: {l2_error:.10e}")

# Plot heatmap over (x,t)
X, T_grid = np.meshgrid(x_fd, t_fd)

plt.figure(figsize=(8, 5))
plt.pcolormesh(X, T_grid, u_fd, shading="auto", cmap="viridis")
plt.colorbar(label="u(x,t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Forward Euler Finite-Difference Solution of Heat Equation")
plt.tight_layout()
plt.savefig("heat_fd_heatmap.png", dpi=300)
plt.show()

# Optional: plot final-time comparison
plt.figure(figsize=(7, 4))
plt.plot(x_fd, u_exact_final, "k-", lw=2, label="Exact at t=0.5")
plt.plot(x_fd, u_fd[-1, :], "r--", lw=1.5, label="FD solution at t=0.5")
plt.xlabel("x")
plt.ylabel("u(x,0.5)")
plt.title("Heat Equation: Finite Difference vs Exact at t=0.5")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("heat_fd_final_comparison.png", dpi=300)
plt.show()