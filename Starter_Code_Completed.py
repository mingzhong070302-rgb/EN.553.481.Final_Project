"""
Starter Code: PINN Final Project
EN 553.481/681 Numerical Analysis
"""
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLLOC_EPOCHS = 1000

class InversePINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Train log(nu), so nu stays positive
        self.log_nu = nn.Parameter(torch.tensor(np.log(0.02), dtype=torch.float32))

    def forward(self, x):
        return self.net(x)

    def nu(self):
        return torch.exp(self.log_nu)


def generate_noisy_heat_data(Ndata=50, sigma=0.01, nu_true=0.01):
    x_data = torch.rand(Ndata, 1, device=device)
    t_data = 0.5 * torch.rand(Ndata, 1, device=device)

    x_np = x_data.cpu().numpy()
    t_np = t_data.cpu().numpy()

    u_true = (
        np.exp(-nu_true * np.pi**2 * t_np) * np.sin(np.pi * x_np)
        + 0.5 * np.exp(-9 * nu_true * np.pi**2 * t_np) * np.sin(3 * np.pi * x_np)
    )

    u_noisy = u_true + sigma * np.random.randn(*u_true.shape)

    xt_data = torch.cat([x_data, t_data], dim=1)
    u_data = torch.tensor(u_noisy, dtype=torch.float32, device=device)

    return xt_data, u_data


def compute_loss_inverse_heat_ad(model, xt_data, u_data):
    Nr = 500
    Nic = 100
    Nbc = 100

    x_r = torch.rand(Nr, 1, device=device)
    t_r = 0.5 * torch.rand(Nr, 1, device=device)

    x_r.requires_grad_(True)
    t_r.requires_grad_(True)

    xt_r = torch.cat([x_r, t_r], dim=1)
    u = model(xt_r)

    u_t = torch.autograd.grad(
        u,
        t_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u,
        x_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x_r,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    Lr = torch.mean((u_t - model.nu() * u_xx)**2)

    x_ic = torch.rand(Nic, 1, device=device)
    t_ic = torch.zeros(Nic, 1, device=device)
    xt_ic = torch.cat([x_ic, t_ic], dim=1)

    u_ic_pred = model(xt_ic)
    u_ic_true = torch.sin(np.pi * x_ic) + 0.5 * torch.sin(3 * np.pi * x_ic)
    Lic = torch.mean((u_ic_pred - u_ic_true)**2)

    t_bc = 0.5 * torch.rand(Nbc, 1, device=device)
    x_left = torch.zeros(Nbc, 1, device=device)
    x_right = torch.ones(Nbc, 1, device=device)

    Lbc = torch.mean(model(torch.cat([x_left, t_bc], dim=1))**2)
    Lbc += torch.mean(model(torch.cat([x_right, t_bc], dim=1))**2)

    Ldata = torch.mean((model(xt_data) - u_data)**2)

    return Lr + 20*Lic + 20*Lbc + 100*Ldata


def compute_loss_inverse_heat_fdm(model, xt_data, u_data, epsilon=1e-3):
    Nr = 500
    Nic = 100
    Nbc = 100

    x_r = epsilon + (1 - 2 * epsilon) * torch.rand(Nr, 1, device=device)
    t_r = epsilon + (0.5 - 2 * epsilon) * torch.rand(Nr, 1, device=device)

    xt = torch.cat([x_r, t_r], dim=1)
    xt_t_plus = torch.cat([x_r, t_r + epsilon], dim=1)
    xt_t_minus = torch.cat([x_r, t_r - epsilon], dim=1)
    xt_x_plus = torch.cat([x_r + epsilon, t_r], dim=1)
    xt_x_minus = torch.cat([x_r - epsilon, t_r], dim=1)

    u = model(xt)

    u_t_fdm = (model(xt_t_plus) - model(xt_t_minus)) / (2 * epsilon)
    u_xx_fdm = (model(xt_x_plus) - 2*u + model(xt_x_minus)) / (epsilon**2)

    Lr = torch.mean((u_t_fdm - model.nu() * u_xx_fdm)**2)

    x_ic = torch.rand(Nic, 1, device=device)
    t_ic = torch.zeros(Nic, 1, device=device)
    xt_ic = torch.cat([x_ic, t_ic], dim=1)

    u_ic_pred = model(xt_ic)
    u_ic_true = torch.sin(np.pi * x_ic) + 0.5 * torch.sin(3 * np.pi * x_ic)
    Lic = torch.mean((u_ic_pred - u_ic_true)**2)

    t_bc = 0.5 * torch.rand(Nbc, 1, device=device)
    x_left = torch.zeros(Nbc, 1, device=device)
    x_right = torch.ones(Nbc, 1, device=device)

    Lbc = torch.mean(model(torch.cat([x_left, t_bc], dim=1))**2)
    Lbc += torch.mean(model(torch.cat([x_right, t_bc], dim=1))**2)

    Ldata = torch.mean((model(xt_data) - u_data)**2)

    return Lr + 20*Lic + 20*Lbc + 100*Ldata


def compute_loss_heat_ad_Nr(model, Nr=10000):
    nu = 0.01

    x_r = torch.rand(Nr, 1, device=device)
    t_r = 0.5 * torch.rand(Nr, 1, device=device)

    x_r.requires_grad_(True)
    t_r.requires_grad_(True)

    xt_r = torch.cat([x_r, t_r], dim=1)
    u = model(xt_r)

    u_t = torch.autograd.grad(
        u,
        t_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u,
        x_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x_r,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    residual = u_t - nu * u_xx
    Lr = torch.mean(residual**2)

    Nic = 200
    x_ic = torch.rand(Nic, 1, device=device)
    t_ic = torch.zeros(Nic, 1, device=device)

    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(xt_ic)
    u_ic_true = torch.sin(np.pi * x_ic) + 0.5 * torch.sin(3 * np.pi * x_ic)

    Lic = torch.mean((u_ic_pred - u_ic_true)**2)

    Nbc = 200
    t_bc = 0.5 * torch.rand(Nbc, 1, device=device)

    x_left = torch.zeros(Nbc, 1, device=device)
    x_right = torch.ones(Nbc, 1, device=device)

    u_left = model(torch.cat([x_left, t_bc], dim=1))
    u_right = model(torch.cat([x_right, t_bc], dim=1))

    Lbc = torch.mean(u_left**2) + torch.mean(u_right**2)

    return Lr + 20*Lic + 20*Lbc


def compute_loss_heat_fdm_Nr(model, Nr=10000, epsilon=1e-3):
    nu = 0.01

    x_r = epsilon + (1 - 2 * epsilon) * torch.rand(Nr, 1, device=device)
    t_r = epsilon + (0.5 - 2 * epsilon) * torch.rand(Nr, 1, device=device)

    xt = torch.cat([x_r, t_r], dim=1)
    xt_t_plus = torch.cat([x_r, t_r + epsilon], dim=1)
    xt_t_minus = torch.cat([x_r, t_r - epsilon], dim=1)
    xt_x_plus = torch.cat([x_r + epsilon, t_r], dim=1)
    xt_x_minus = torch.cat([x_r - epsilon, t_r], dim=1)

    u = model(xt)

    u_t_fdm = (model(xt_t_plus) - model(xt_t_minus)) / (2 * epsilon)
    u_xx_fdm = (model(xt_x_plus) - 2*u + model(xt_x_minus)) / (epsilon**2)

    residual = u_t_fdm - nu * u_xx_fdm
    Lr = torch.mean(residual**2)

    Nic = 200
    x_ic = torch.rand(Nic, 1, device=device)
    t_ic = torch.zeros(Nic, 1, device=device)

    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(xt_ic)
    u_ic_true = torch.sin(np.pi * x_ic) + 0.5 * torch.sin(3 * np.pi * x_ic)

    Lic = torch.mean((u_ic_pred - u_ic_true)**2)

    Nbc = 200
    t_bc = 0.5 * torch.rand(Nbc, 1, device=device)

    x_left = torch.zeros(Nbc, 1, device=device)
    x_right = torch.ones(Nbc, 1, device=device)

    u_left = model(torch.cat([x_left, t_bc], dim=1))
    u_right = model(torch.cat([x_right, t_bc], dim=1))

    Lbc = torch.mean(u_left**2) + torch.mean(u_right**2)

    return Lr + 20*Lic + 20*Lbc


def compute_ode_max_error(model, exact_fn, t_range=(0, 5)):
    t = torch.linspace(*t_range, 1000, device=device).unsqueeze(1)
    with torch.no_grad():
        u_pred = model(t).cpu().numpy().flatten()
    t_np = t.cpu().numpy().flatten()
    u_ex = exact_fn(t_np)
    return np.max(np.abs(u_pred - u_ex))

# Determine fast or slow train mode
FAST_TEST = False
if FAST_TEST:
    HEAT_EPOCHS = 1000
    HEAT_NR = 1000
else:
    HEAT_EPOCHS = 20000
    HEAT_NR = 10000
    
# New lost funciton just for Problem 3
def compute_loss_ode_ad_Nr(model, Nr=500):
    t_r = 5 * torch.rand(Nr, 1, device=device)
    t_r.requires_grad_(True)

    u = model(t_r)

    du_dt = torch.autograd.grad(
        u, t_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    residual = du_dt + 5*u - 5*torch.cos(t_r) + torch.sin(t_r)
    Lr = torch.mean(residual**2)

    t0 = torch.tensor([[0.0]], device=device)
    Lic = torch.mean(model(t0)**2)

    return Lr + 50*Lic


def compute_loss_ode_fdm_Nr(model, Nr=500, epsilon=1e-3):
    t_r = 5 * torch.rand(Nr, 1, device=device)

    u = model(t_r)
    u_plus = model(t_r + epsilon)
    u_minus = model(t_r - epsilon)

    du_dt_fdm = (u_plus - u_minus) / (2 * epsilon)

    residual = du_dt_fdm + 5*u - 5*torch.cos(t_r) + torch.sin(t_r)
    Lr = torch.mean(residual**2)

    t0 = torch.tensor([[0.0]], device=device)
    Lic = torch.mean(model(t0)**2)

    return Lr + 50*Lic

# Helper functon to compute the relative L^2 error without making extra plots
def compute_heat_relative_l2_error(model, exact_fn):
    Ntest = 100
    x = np.linspace(0, 1, Ntest)
    t = np.linspace(0, 0.5, Ntest)
    X, T = np.meshgrid(x, t)

    xt = np.column_stack([X.ravel(), T.ravel()])
    xt_t = torch.tensor(xt, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = model(xt_t).cpu().numpy().reshape(Ntest, Ntest)

    u_ex = exact_fn(X, T)

    rel_l2 = np.sqrt(np.sum((u_pred - u_ex)**2)) / np.sqrt(np.sum(u_ex**2))
    return rel_l2

# Helper function for foward euler for heat equations
def heat_forward_euler(dx=1/64, r=0.4, T=0.5, nu=0.01):
    # Compute dt from r = nu * dt / dx^2
    dt = r * dx**2 / nu

    # Spatial grid
    Nx = int(1 / dx)
    x = np.linspace(0, 1, Nx + 1)

    # Time grid
    Nt = int(np.ceil(T / dt))
    dt = T / Nt
    r = nu * dt / dx**2
    t = np.linspace(0, T, Nt + 1)

    # Solution array: rows are time, columns are space
    u = np.zeros((Nt + 1, Nx + 1))

    # Initial condition
    u[0, :] = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)

    # Boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0

    # Forward Euler update
    for n in range(Nt):
        for i in range(1, Nx):
            u[n+1, i] = u[n, i] + r * (
                u[n, i+1] - 2*u[n, i] + u[n, i-1]
            )

    return x, t, u, dt, r

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_pinn(model, loss_fn, epochs, lr=1e-3, log_every=2000):
    """Train a PINN model.
    Returns: (loss_history, wall_clock_time_seconds)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % log_every == 0:
            print(f"  Epoch {epoch}/{epochs}, Loss = {loss.item():.6e}")
    wall_time = time.time() - t_start
    print(f"  Training time: {wall_time:.1f}s")
    return loss_history, wall_time

def plot_loss_curve(loss_history, title="Training Loss"):
    plt.figure(figsize=(6, 4))
    plt.semilogy(loss_history)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(title); plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_ode_comparison(model, exact_fn, t_range=(0, 5), label="PINN"):
    t = torch.linspace(*t_range, 1000, device=device).unsqueeze(1)
    with torch.no_grad():
        u_pred = model(t).cpu().numpy().flatten()
    t_np = t.cpu().numpy().flatten()
    u_ex = exact_fn(t_np)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_np, u_ex, 'k-', lw=2, label='Exact')
    axes[0].plot(t_np, u_pred, 'r--', lw=1.5, label=label)
    axes[0].set_xlabel('t'); axes[0].set_ylabel('u(t)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{label} vs Exact')

    err = np.abs(u_pred - u_ex)
    axes[1].plot(t_np, err, 'b-')
    axes[1].set_xlabel('t'); axes[1].set_ylabel('|error|')
    axes[1].set_title(f'Pointwise Error (max = {err.max():.4e})')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    print(f"  Max absolute error: {err.max():.6e}")
    return err.max()

def plot_heat_comparison(model, exact_fn, label="PINN"):
    """Plot PINN vs exact for heat eq. Returns relative L2 error."""
    Ntest = 100
    x = np.linspace(0, 1, Ntest)
    t = np.linspace(0, 0.5, Ntest)
    X, T = np.meshgrid(x, t)
    xt = np.column_stack([X.ravel(), T.ravel()])
    xt_t = torch.tensor(xt, dtype=torch.float32, device=device)
    with torch.no_grad():
        u_pred = model(xt_t).cpu().numpy().reshape(Ntest, Ntest)
    u_ex = exact_fn(X, T)
    err = np.abs(u_pred - u_ex)
    rel_l2 = np.sqrt(np.sum((u_pred - u_ex)**2)) / np.sqrt(np.sum(u_ex**2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    c0 = axes[0].pcolormesh(X, T, u_pred, shading='auto', cmap='viridis')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('t')
    axes[0].set_title(f'{label} Prediction'); plt.colorbar(c0, ax=axes[0])
    c1 = axes[1].pcolormesh(X, T, u_ex, shading='auto', cmap='viridis')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('t')
    axes[1].set_title('Exact Solution'); plt.colorbar(c1, ax=axes[1])
    c2 = axes[2].pcolormesh(X, T, err, shading='auto', cmap='hot')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('t')
    axes[2].set_title(f'|Error| (rel L2 = {rel_l2:.4e})'); plt.colorbar(c2, ax=axes[2])
    plt.tight_layout()
    print(f"  Relative L2 error: {rel_l2:.6e}")
    return rel_l2

# =============================================================
# TODO: Implement these four loss functions
# =============================================================

def compute_loss_ode_ad(model):
    """PINN loss for ODE using AUTOGRAD.

    ODE: du/dt = -5u + 5cos(t) - sin(t),  u(0) = 0
    """

    # (i) Sample Nr = 500 collocation points uniformly from [0, 5]
    Nr = 500
    t_r = 5 * torch.rand(Nr, 1, device=device)
    t_r.requires_grad_(True)

    # Network prediction u_theta(t)
    u = model(t_r)

    # (ii) Compute du_theta/dt using automatic differentiation
    du_dt = torch.autograd.grad(
        u,
        t_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # (iii) Residual:
    # du/dt + 5u - 5cos(t) + sin(t) = 0
    residual = du_dt + 5*u - 5*torch.cos(t_r) + torch.sin(t_r)
    Lr = torch.mean(residual**2)

    # (iv) Initial condition loss: |u_theta(0)|^2
    t0 = torch.tensor([[0.0]], device=device)
    u0_pred = model(t0)
    Lic = torch.mean(u0_pred**2)

    # (v) Return total loss
    return Lr + 50*Lic

def compute_loss_ode_fdm(model, epsilon=1e-3):
    """PINN loss for ODE using FINITE DIFFERENCES.

    ODE: du/dt = -5u + 5cos(t) - sin(t),  u(0) = 0
    """

    # Sample Nr = 500 collocation points uniformly from [0, 5]
    Nr = 500
    t_r = 5 * torch.rand(Nr, 1, device=device)

    # Network evaluations for central difference
    u_plus = model(t_r + epsilon)
    u_minus = model(t_r - epsilon)
    u = model(t_r)

    # Central difference approximation:
    # du/dt(t) ≈ (u(t+eps) - u(t-eps)) / (2eps)
    du_dt_fdm = (u_plus - u_minus) / (2 * epsilon)

    # Residual:
    # du/dt + 5u - 5cos(t) + sin(t) = 0
    residual = du_dt_fdm + 5*u - 5*torch.cos(t_r) + torch.sin(t_r)
    Lr = torch.mean(residual**2)

    # Initial condition loss: |u_theta(0)|^2
    t0 = torch.tensor([[0.0]], device=device)
    u0_pred = model(t0)
    Lic = torch.mean(u0_pred**2)

    # Total loss
    return Lr + 50*Lic

def compute_loss_heat_ad(model):
    """PINN loss for heat equation using AUTOGRAD.

    PDE: u_t = 0.01 * u_xx  on (0,1) x (0, 0.5]
    IC:  u(x, 0) = sin(pi*x) + 0.5*sin(3*pi*x)
    BC:  u(0, t) = u(1, t) = 0
    """

    nu = 0.01

    # -------------------------
    # PDE residual loss
    # -------------------------
    Nr = HEAT_NR

    x_r = torch.rand(Nr, 1, device=device)
    t_r = 0.5 * torch.rand(Nr, 1, device=device)

    x_r.requires_grad_(True)
    t_r.requires_grad_(True)

    xt_r = torch.cat([x_r, t_r], dim=1)
    u = model(xt_r)

    # u_t
    u_t = torch.autograd.grad(
        u,
        t_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # u_x
    u_x = torch.autograd.grad(
        u,
        x_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # u_xx
    u_xx = torch.autograd.grad(
        u_x,
        x_r,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    residual = u_t - nu * u_xx
    Lr = torch.mean(residual**2)

    # -------------------------
    # Initial condition loss
    # -------------------------
    Nic = 200

    x_ic = torch.rand(Nic, 1, device=device)
    t_ic = torch.zeros(Nic, 1, device=device)

    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(xt_ic)

    u_ic_true = torch.sin(np.pi * x_ic) + 0.5 * torch.sin(3 * np.pi * x_ic)

    Lic = torch.mean((u_ic_pred - u_ic_true)**2)

    # -------------------------
    # Boundary condition loss
    # -------------------------
    Nbc = 200

    t_bc = 0.5 * torch.rand(Nbc, 1, device=device)

    x_left = torch.zeros(Nbc, 1, device=device)
    x_right = torch.ones(Nbc, 1, device=device)

    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)

    u_left = model(xt_left)
    u_right = model(xt_right)

    Lbc = torch.mean(u_left**2) + torch.mean(u_right**2)

    return Lr + 20*Lic + 20*Lbc

def compute_loss_heat_fdm(model, epsilon=1e-3):
    """PINN loss for heat equation using FINITE DIFFERENCES.

    PDE: u_t = 0.01 * u_xx  on (0,1) x (0, 0.5]
    IC:  u(x, 0) = sin(pi*x) + 0.5*sin(3*pi*x)
    BC:  u(0, t) = u(1, t) = 0
    """

    nu = 0.01

    # -------------------------
    # PDE residual loss
    # -------------------------
    Nr = HEAT_NR

    # To avoid evaluating outside the domain when using t +/- epsilon and x +/- epsilon,
    # sample away from the boundary.
    x_r = epsilon + (1 - 2 * epsilon) * torch.rand(Nr, 1, device=device)
    t_r = epsilon + (0.5 - 2 * epsilon) * torch.rand(Nr, 1, device=device)

    xt = torch.cat([x_r, t_r], dim=1)
    xt_t_plus = torch.cat([x_r, t_r + epsilon], dim=1)
    xt_t_minus = torch.cat([x_r, t_r - epsilon], dim=1)

    xt_x_plus = torch.cat([x_r + epsilon, t_r], dim=1)
    xt_x_minus = torch.cat([x_r - epsilon, t_r], dim=1)

    u = model(xt)
    u_t_plus = model(xt_t_plus)
    u_t_minus = model(xt_t_minus)

    u_x_plus = model(xt_x_plus)
    u_x_minus = model(xt_x_minus)

    # Central differences
    u_t_fdm = (u_t_plus - u_t_minus) / (2 * epsilon)
    u_xx_fdm = (u_x_plus - 2 * u + u_x_minus) / (epsilon**2)

    residual = u_t_fdm - nu * u_xx_fdm
    Lr = torch.mean(residual**2)

    # -------------------------
    # Initial condition loss
    # -------------------------
    Nic = 200

    x_ic = torch.rand(Nic, 1, device=device)
    t_ic = torch.zeros(Nic, 1, device=device)

    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(xt_ic)

    u_ic_true = torch.sin(np.pi * x_ic) + 0.5 * torch.sin(3 * np.pi * x_ic)

    Lic = torch.mean((u_ic_pred - u_ic_true)**2)

    # -------------------------
    # Boundary condition loss
    # -------------------------
    Nbc = 200

    t_bc = 0.5 * torch.rand(Nbc, 1, device=device)

    x_left = torch.zeros(Nbc, 1, device=device)
    x_right = torch.ones(Nbc, 1, device=device)

    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)

    u_left = model(xt_left)
    u_right = model(xt_right)

    Lbc = torch.mean(u_left**2) + torch.mean(u_right**2)

    return Lr + 20*Lic + 20*Lbc

if __name__ == "__main__":
    ode_exact = lambda t: np.cos(t) - np.exp(-5*t)
    nu = 0.01
    heat_exact = lambda x, t: (
    np.exp(-nu * np.pi**2 * t) * np.sin(np.pi * x)
    + 0.5 * np.exp(-9 * nu * np.pi**2 * t) * np.sin(3 * np.pi * x))

    # --- Problem 1.2: ODE with AD ---
    print("=" * 50)
    print("Problem 1.2: ODE PINN (Autograd)")
    print("=" * 50)

    model_ode_ad = PINN(
    input_dim=1,
    hidden_dim=32,
    num_layers=3,
    output_dim=1).to(device)

    loss_ode_ad, time_ode_ad = train_pinn(
    model_ode_ad,
    compute_loss_ode_ad,
    epochs=10000,
    lr=1e-3,
    log_every=2000)

    plot_loss_curve(loss_ode_ad, title="ODE PINN with Autograd: Training Loss")
    plt.savefig("ode_ad_loss.png", dpi=300)

    error_ode_ad = plot_ode_comparison(
    model_ode_ad,
    ode_exact,
    t_range=(0, 5),
    label="ODE AD-PINN")
    plt.savefig("ode_ad_solution.png", dpi=300)
    plt.show()

    # --- Problem 1.3: ODE with FDM ---
    print("\n" + "=" * 50)
    print("Problem 1.3: ODE PINN (FDM)")
    print("=" * 50)

    model_ode_fdm = PINN(
    input_dim=1,
    hidden_dim=32,
    num_layers=3,
    output_dim=1).to(device)

    loss_ode_fdm, time_ode_fdm = train_pinn(
    model_ode_fdm,
    lambda model: compute_loss_ode_fdm(model, epsilon=1e-3),
    epochs=10000,
    lr=1e-3,
    log_every=2000)

    plot_loss_curve(loss_ode_fdm, title="ODE PINN with FDM: Training Loss")
    plt.savefig("ode_fdm_loss.png", dpi=300)

    error_ode_fdm = plot_ode_comparison(
    model_ode_fdm,
    ode_exact,
    t_range=(0, 5),
    label="ODE FDM-PINN")
    plt.savefig("ode_fdm_solution.png", dpi=300)
    plt.show()
    
    # --- Problem 1.4(a): Compare AD-PINN and FDM-PINN ---
    print("\n" + "=" * 70)
    print("Problem 1.4(a): ODE PINN Comparison")
    print("=" * 70)
    print(f"{'Method':<15}{'Final Training Loss':<25}{'Max Abs Error':<20}{'Time (s)':<15}")
    print("-" * 70)

    print(f"{'AD-PINN':<15}{loss_ode_ad[-1]:<25.10e}{error_ode_ad:<20.10e}{time_ode_ad:<15.4f}")
    print(f"{'FDM-PINN':<15}{loss_ode_fdm[-1]:<25.10e}{error_ode_fdm:<20.10e}{time_ode_fdm:<15.4f}")
    
    # --- Problem 1.4(b): Effect of epsilon on FDM-PINN ---
    print("\n" + "=" * 70)
    print("Problem 1.4(b): FDM-PINN Epsilon Study")
    print("=" * 70)

    eps_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    eps_errors = []
    eps_losses = []
    eps_times = []

    for eps in eps_values:
        print(f"\nTraining FDM-PINN with epsilon = {eps:.0e}")

        model_eps = PINN(
        input_dim=1,
        hidden_dim=32,
        num_layers=3,
        output_dim=1).to(device)

        loss_eps, time_eps = train_pinn(
        model_eps,
        lambda model, eps=eps: compute_loss_ode_fdm(model, epsilon=eps),
        epochs=10000,
        lr=1e-3,
        log_every=2000)

        error_eps = plot_ode_comparison(
        model_eps,
        ode_exact,
        t_range=(0, 5),
        label=f"FDM-PINN eps={eps:.0e}")
        plt.savefig(f"ode_fdm_eps_{eps:.0e}_solution.png", dpi=300)

        eps_errors.append(error_eps)
        eps_losses.append(loss_eps[-1])
        eps_times.append(time_eps)

    # Print epsilon study table
    print("\nFDM-PINN Epsilon Study")
    print("-" * 70)
    print(f"{'epsilon':<15}{'Final Training Loss':<25}{'Max Abs Error':<20}{'Time (s)':<15}")
    print("-" * 70)

    for eps, loss_val, err_val, time_val in zip(eps_values, eps_losses, eps_errors, eps_times):
        print(f"{eps:<15.0e}{loss_val:<25.10e}{err_val:<20.10e}{time_val:<15.4f}")

    # Plot max error vs epsilon
    plt.figure(figsize=(6, 4))
    plt.loglog(eps_values, eps_errors, marker="o")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Max absolute error")
    plt.title("FDM-PINN Error vs Finite-Difference Step Size")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("ode_fdm_error_vs_epsilon.png", dpi=300)
    plt.show()

    # --- Problem 2.2: Heat with AD ---
    print("\n" + "=" * 50)
    print("Problem 2.2: Heat PINN (Autograd)")
    print("=" * 50)
    model_heat_ad = PINN(
    input_dim=2,
    hidden_dim=64,
    num_layers=4,
    output_dim=1).to(device)

    loss_heat_ad, time_heat_ad = train_pinn(
    model_heat_ad,
    compute_loss_heat_ad,
    epochs=HEAT_EPOCHS,
    lr=1e-3,
    log_every=2000)

    plot_loss_curve(loss_heat_ad, title="Heat PINN with Autograd: Training Loss")
    plt.savefig("heat_ad_loss.png", dpi=300)

    error_heat_ad = plot_heat_comparison(
    model_heat_ad,
    heat_exact,
    label="Heat AD-PINN")
    plt.savefig("heat_ad_solution_error.png", dpi=300)
    plt.show()

    # --- Problem 2.3: Heat with FDM ---
    print("\n" + "=" * 50)
    print("Problem 2.3: Heat PINN (FDM)")
    print("=" * 50)
    model_heat_fdm = PINN(
    input_dim=2,
    hidden_dim=64,
    num_layers=4,
    output_dim=1).to(device)

    loss_heat_fdm, time_heat_fdm = train_pinn(
    model_heat_fdm,
    lambda model: compute_loss_heat_fdm(model, epsilon=1e-3),
    epochs=HEAT_EPOCHS,
    lr=1e-3,
    log_every=2000)

    plot_loss_curve(loss_heat_fdm, title="Heat PINN with FDM: Training Loss")
    plt.savefig("heat_fdm_loss.png", dpi=300)

    error_heat_fdm = plot_heat_comparison(
    model_heat_fdm,
    heat_exact,
    label="Heat FDM-PINN")
    plt.savefig("heat_fdm_solution_error.png", dpi=300)
    plt.show()
    
    # --- Problem 2.4(a): Compare Heat AD-PINN and FDM-PINN ---
    print("\n" + "=" * 70)
    print("Problem 2.4(a): Heat PINN Comparison")
    print("=" * 70)
    print(f"{'Method':<15}{'Final Loss':<25}{'Relative L2 Error':<25}{'Time (s)':<15}")
    print("-" * 80)

    print(f"{'AD-PINN':<15}{loss_heat_ad[-1]:<25.10e}{error_heat_ad:<25.10e}{time_heat_ad:<15.4f}")
    print(f"{'FDM-PINN':<15}{loss_heat_fdm[-1]:<25.10e}{error_heat_fdm:<25.10e}{time_heat_fdm:<15.4f}")
    
    # --- Problem 2.4(b): Heat FDM-PINN epsilon study ---
    print("\n" + "=" * 70)
    print("Problem 2.4(b): Heat FDM-PINN Epsilon Study")
    print("=" * 70)

    heat_eps_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    heat_eps_errors = []
    heat_eps_losses = []
    heat_eps_times = []

    for eps in heat_eps_values:
        print(f"\nTraining Heat FDM-PINN with epsilon = {eps:.0e}")

        model_heat_eps = PINN(
        input_dim=2,
        hidden_dim=64,
        num_layers=4,
        output_dim=1).to(device)

        loss_eps, time_eps = train_pinn(
        model_heat_eps,
        lambda model, eps=eps: compute_loss_heat_fdm(model, epsilon=eps),
        epochs=HEAT_EPOCHS,
        lr=1e-3,
        log_every=2000)

        rel_l2_eps = compute_heat_relative_l2_error(model_heat_eps, heat_exact)

        heat_eps_errors.append(rel_l2_eps)
        heat_eps_losses.append(loss_eps[-1])
        heat_eps_times.append(time_eps)

    print("\nHeat FDM-PINN Epsilon Study")
    print("-" * 80)
    print(f"{'epsilon':<15}{'Final Loss':<25}{'Relative L2 Error':<25}{'Time (s)':<15}")
    print("-" * 80)

    for eps, loss_val, err_val, time_val in zip(
            heat_eps_values, heat_eps_losses, heat_eps_errors, heat_eps_times):
        print(f"{eps:<15.0e}{loss_val:<25.10e}{err_val:<25.10e}{time_val:<15.4f}")

    best_index = np.argmin(heat_eps_errors)
    best_eps = heat_eps_values[best_index]
    best_error = heat_eps_errors[best_index]

    print(f"\nBest epsilon among tested values: {best_eps:.0e}")
    print(f"Smallest relative L2 error: {best_error:.10e}")

    plt.figure(figsize=(6, 4))
    plt.loglog(heat_eps_values, heat_eps_errors, marker="o")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"Relative $L^2$ error")
    plt.title(r"Heat FDM-PINN Relative $L^2$ Error vs. $\epsilon$")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("heat_fdm_error_vs_epsilon.png", dpi=300)
    plt.show()
    
    # --- Problem 2.4(c): Unstable Forward Euler FD with r = 0.6 ---
    print("\n" + "=" * 70)
    print("Problem 2.4(c): Forward Euler FD with CFL Violation")
    print("=" * 70)

    x_bad, t_bad, u_bad, dt_bad, r_bad = heat_forward_euler(
    dx=1/64,
    r=0.6,
    T=0.5,
    nu=0.01)

    print(f"Delta x = {1/64:.10e}")
    print(f"Delta t = {dt_bad:.10e}")
    print(f"r = {r_bad:.10e}")
    print(f"Max |u| = {np.max(np.abs(u_bad)):.10e}")

    X_bad, T_bad = np.meshgrid(x_bad, t_bad)

    plt.figure(figsize=(8, 5))
    plt.pcolormesh(X_bad, T_bad, u_bad, shading="auto", cmap="viridis")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Unstable Forward Euler FD Heat Solution with r = 0.6")
    plt.tight_layout()
    plt.savefig("heat_fd_unstable_r06.png", dpi=300)
    plt.show()
    
    # --- Problem 3(a): Error Comparison Summary --- We input values from above
    # so we don't need to rerun training (which takes a lot of time!)
    print("\n" + "=" * 90)
    print("Problem 3(a): Error Comparison Summary")
    print("=" * 90)
    print(f"{'Method':<25}{'Problem':<15}{'Error Metric':<25}{'Error':<18}{'Time (s)':<12}")
    print("-" * 90)

    print(f"{'Forward Euler':<25}{'ODE':<15}{'Max abs error':<25}{1.0029780605e-02:<18.10e}{'-':<12}")
    print(f"{'RK4':<25}{'ODE':<15}{'Max abs error':<25}{2.1646788473e-08:<18.10e}{'-':<12}")
    print(f"{'Forward Euler FD':<25}{'Heat':<15}{'L2 error at t=0.5':<25}{2.4914748483e-04:<18.10e}{'-':<12}")

    print(f"{'AD-PINN':<25}{'ODE':<15}{'Max abs error':<25}{1.3652941212e-02:<18.10e}{5.7595:<12.4f}")
    print(f"{'FDM-PINN':<25}{'ODE':<15}{'Max abs error':<25}{3.0182003975e-03:<18.10e}{8.7307:<12.4f}")
    
    print(f"{'AD-PINN':<25}{'Heat':<15}{'Relative L2 error':<25}{7.1848615855e-03:<18.10e}{788.7079:<12.4f}")
    print(f"{'FDM-PINN':<25}{'Heat':<15}{'Relative L2 error':<25}{9.3526164268e-03:<18.10e}{603.2352:<12.4f}")

    # --- Problem 3(b): Effect of collocation points ---
    print("\n" + "=" * 90)
    print("Problem 3(b): Effect of Collocation Points")
    print("=" * 90)

    # Use shorter training for the ablation study so it runs faster.
    COLLOC_EPOCHS = 1000
    LOG_EVERY_COLLOC = 500

    # -------------------------
    # ODE collocation study
    # -------------------------
    print("\nODE Collocation Point Study")
    print("-" * 70)

    Nr_values_ode = [100, 500, 2000, 10000]

    ode_ad_Nr_errors = []
    ode_fdm_Nr_errors = []

    for Nr in Nr_values_ode:
        print(f"\nODE AD-PINN with Nr = {Nr}")

        model_ode_ad_Nr = PINN(
            input_dim=1,
            hidden_dim=32,
            num_layers=3,
            output_dim=1
        ).to(device)

        train_pinn(
            model_ode_ad_Nr,
            lambda model, Nr=Nr: compute_loss_ode_ad_Nr(model, Nr=Nr),
            epochs=COLLOC_EPOCHS,
            lr=1e-3,
            log_every=LOG_EVERY_COLLOC
        )

        err_ad = compute_ode_max_error(model_ode_ad_Nr, ode_exact)
        ode_ad_Nr_errors.append(err_ad)

        print(f"ODE FDM-PINN with Nr = {Nr}")

        model_ode_fdm_Nr = PINN(
            input_dim=1,
            hidden_dim=32,
            num_layers=3,
            output_dim=1
        ).to(device)

        train_pinn(
            model_ode_fdm_Nr,
            lambda model, Nr=Nr: compute_loss_ode_fdm_Nr(model, Nr=Nr, epsilon=1e-3),
            epochs=COLLOC_EPOCHS,
            lr=1e-3,
            log_every=LOG_EVERY_COLLOC
        )

        err_fdm = compute_ode_max_error(model_ode_fdm_Nr, ode_exact)
        ode_fdm_Nr_errors.append(err_fdm)

    print("\nODE Collocation Study Results")
    print("-" * 70)
    print(f"{'Nr':<15}{'AD-PINN Error':<25}{'FDM-PINN Error':<25}")
    print("-" * 70)

    for Nr, err_ad, err_fdm in zip(Nr_values_ode, ode_ad_Nr_errors, ode_fdm_Nr_errors):
        print(f"{Nr:<15}{err_ad:<25.10e}{err_fdm:<25.10e}")

    plt.figure(figsize=(6, 4))
    plt.loglog(Nr_values_ode, ode_ad_Nr_errors, marker="o", label="AD-PINN")
    plt.loglog(Nr_values_ode, ode_fdm_Nr_errors, marker="o", label="FDM-PINN")
    plt.xlabel(r"$N_r$")
    plt.ylabel("Max absolute error")
    plt.title(r"ODE Error vs. Number of Collocation Points")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("ode_error_vs_Nr.png", dpi=300)
    plt.show()

    # -------------------------
    # Heat equation collocation study
    # -------------------------
    print("\nHeat Equation Collocation Point Study")
    print("-" * 70)

    Nr_values_heat = [500, 2000, 10000]

    heat_ad_Nr_errors = []
    heat_fdm_Nr_errors = []

    for Nr in Nr_values_heat:
        print(f"\nHeat AD-PINN with Nr = {Nr}")

        model_heat_ad_Nr = PINN(
            input_dim=2,
            hidden_dim=64,
            num_layers=4,
            output_dim=1
        ).to(device)

        train_pinn(
            model_heat_ad_Nr,
            lambda model, Nr=Nr: compute_loss_heat_ad_Nr(model, Nr=Nr),
            epochs=COLLOC_EPOCHS,
            lr=1e-3,
            log_every=LOG_EVERY_COLLOC
        )

        err_ad = compute_heat_relative_l2_error(model_heat_ad_Nr, heat_exact)
        heat_ad_Nr_errors.append(err_ad)

        print(f"Heat FDM-PINN with Nr = {Nr}")

        model_heat_fdm_Nr = PINN(
            input_dim=2,
            hidden_dim=64,
            num_layers=4,
            output_dim=1
        ).to(device)

        train_pinn(
            model_heat_fdm_Nr,
            lambda model, Nr=Nr: compute_loss_heat_fdm_Nr(model, Nr=Nr, epsilon=1e-3),
            epochs=COLLOC_EPOCHS,
            lr=1e-3,
            log_every=LOG_EVERY_COLLOC
        )

        err_fdm = compute_heat_relative_l2_error(model_heat_fdm_Nr, heat_exact)
        heat_fdm_Nr_errors.append(err_fdm)

    print("\nHeat Collocation Study Results")
    print("-" * 70)
    print(f"{'Nr':<15}{'AD-PINN Rel L2':<25}{'FDM-PINN Rel L2':<25}")
    print("-" * 70)

    for Nr, err_ad, err_fdm in zip(Nr_values_heat, heat_ad_Nr_errors, heat_fdm_Nr_errors):
        print(f"{Nr:<15}{err_ad:<25.10e}{err_fdm:<25.10e}")

    plt.figure(figsize=(6, 4))
    plt.loglog(Nr_values_heat, heat_ad_Nr_errors, marker="o", label="AD-PINN")
    plt.loglog(Nr_values_heat, heat_fdm_Nr_errors, marker="o", label="FDM-PINN")
    plt.xlabel(r"$N_r$")
    plt.ylabel(r"Relative $L^2$ error")
    plt.title(r"Heat Equation Error vs. Number of Collocation Points")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("heat_error_vs_Nr.png", dpi=300)
    plt.show()

    # --- Problem 3(c): Effect of network size ---
    print("\n" + "=" * 90)
    print("Problem 3(c): Effect of Network Size")
    print("=" * 90)

    SIZE_EPOCHS = 1000
    LOG_EVERY_SIZE = 500

    network_configs = [
    ("Small", 16, 2),
    ("Large", 64, 5)
    ]

    # Store results as tuples:
    # (Problem, Method, Network, Error Metric, Error, Time)
    size_results = []

    # -------------------------
    # ODE network size study
    # -------------------------
    for net_name, hidden_dim, num_layers in network_configs:
        print(f"\nODE AD-PINN with {net_name} network: {num_layers} layers, {hidden_dim} neurons")

        model_ode_ad_size = PINN(
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1
        ).to(device)

        loss_tmp, time_tmp = train_pinn(
        model_ode_ad_size,
        lambda model: compute_loss_ode_ad_Nr(model, Nr=500),
        epochs=SIZE_EPOCHS,
        lr=1e-3,
        log_every=LOG_EVERY_SIZE
    )

    err_tmp = compute_ode_max_error(model_ode_ad_size, ode_exact)

    size_results.append((
        "ODE",
        "AD-PINN",
        f"{net_name} ({num_layers} layers, {hidden_dim} neurons)",
        "Max abs error",
        err_tmp,
        time_tmp
    ))

    print(f"\nODE FDM-PINN with {net_name} network: {num_layers} layers, {hidden_dim} neurons")

    model_ode_fdm_size = PINN(
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1
    ).to(device)

    loss_tmp, time_tmp = train_pinn(
        model_ode_fdm_size,
        lambda model: compute_loss_ode_fdm_Nr(model, Nr=500, epsilon=1e-3),
        epochs=SIZE_EPOCHS,
        lr=1e-3,
        log_every=LOG_EVERY_SIZE
    )

    err_tmp = compute_ode_max_error(model_ode_fdm_size, ode_exact)

    size_results.append((
        "ODE",
        "FDM-PINN",
        f"{net_name} ({num_layers} layers, {hidden_dim} neurons)",
        "Max abs error",
        err_tmp,
        time_tmp
    ))

    # -------------------------
    # Heat equation network size study
    # -------------------------
    for net_name, hidden_dim, num_layers in network_configs:
        print(f"\nHeat AD-PINN with {net_name} network: {num_layers} layers, {hidden_dim} neurons")
        
        model_heat_ad_size = PINN(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1
        ).to(device)

        loss_tmp, time_tmp = train_pinn(
        model_heat_ad_size,
        lambda model: compute_loss_heat_ad_Nr(model, Nr=2000),
        epochs=SIZE_EPOCHS,
        lr=1e-3,
        log_every=LOG_EVERY_SIZE
        )

        err_tmp = compute_heat_relative_l2_error(model_heat_ad_size, heat_exact)

        size_results.append((
        "Heat",
        "AD-PINN",
        f"{net_name} ({num_layers} layers, {hidden_dim} neurons)",
        "Relative L2 error",
        err_tmp,
        time_tmp
        ))

        print(f"\nHeat FDM-PINN with {net_name} network: {num_layers} layers, {hidden_dim} neurons")

        model_heat_fdm_size = PINN(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1
        ).to(device)

        loss_tmp, time_tmp = train_pinn(
        model_heat_fdm_size,
        lambda model: compute_loss_heat_fdm_Nr(model, Nr=2000, epsilon=1e-3),
        epochs=SIZE_EPOCHS,
        lr=1e-3,
        log_every=LOG_EVERY_SIZE
        )

        err_tmp = compute_heat_relative_l2_error(model_heat_fdm_size, heat_exact)

        size_results.append((
        "Heat",
        "FDM-PINN",
        f"{net_name} ({num_layers} layers, {hidden_dim} neurons)",
        "Relative L2 error",
        err_tmp,
        time_tmp
        ))

    # Print final table
    print("\nNetwork Size Study Results")
    print("-" * 120)
    print(f"{'Problem':<10}{'Method':<15}{'Network':<35}{'Error Metric':<25}{'Error':<18}{'Time (s)':<12}")
    print("-" * 120)

    for problem, method, network, metric, error, train_time in size_results:
        print(f"{problem:<10}{method:<15}{network:<35}{metric:<25}{error:<18.10e}{train_time:<12.4f}")
    
    # --- Bonus: Inverse Problem for Heat Equation ---
    print("\n" + "=" * 90)
    print("Bonus: Inverse Problem for Heat Equation")
    print("=" * 90)

    nu_true = 0.01
    xt_data, u_data = generate_noisy_heat_data(
    Ndata=50,
    sigma=0.01,
    nu_true=nu_true
    )

    INVERSE_EPOCHS = 500

    # AD inverse PINN
    print("\nInverse Heat AD-PINN")

    model_inverse_ad = InversePINN(
    input_dim=2,
    hidden_dim=32,
    num_layers=3,
    output_dim=1
    ).to(device)

    loss_inverse_ad, time_inverse_ad = train_pinn(
    model_inverse_ad,
    lambda model: compute_loss_inverse_heat_ad(model, xt_data, u_data),
    epochs=INVERSE_EPOCHS,
    lr=1e-3,
    log_every=100
    )

    nu_ad = model_inverse_ad.nu().item()
    rel_err_ad = abs(nu_ad - nu_true) / nu_true

    print(f"Recovered nu using AD-PINN: {nu_ad:.10e}")
    print(f"Relative parameter error using AD-PINN: {rel_err_ad:.10e}")

    # FDM inverse PINN
    print("\nInverse Heat FDM-PINN")

    model_inverse_fdm = InversePINN(
    input_dim=2,
    hidden_dim=32,
    num_layers=3,
    output_dim=1
    ).to(device)

    loss_inverse_fdm, time_inverse_fdm = train_pinn(
    model_inverse_fdm,
    lambda model: compute_loss_inverse_heat_fdm(model, xt_data, u_data, epsilon=1e-3),
    epochs=INVERSE_EPOCHS,
    lr=1e-3,
    log_every=100
)

    nu_fdm = model_inverse_fdm.nu().item()
    rel_err_fdm = abs(nu_fdm - nu_true) / nu_true

    print(f"Recovered nu using FDM-PINN: {nu_fdm:.10e}")
    print(f"Relative parameter error using FDM-PINN: {rel_err_fdm:.10e}")

    print("\nBonus Summary")
    print("-" * 80)
    print(f"{'Method':<20}{'Recovered nu':<25}{'Relative error':<25}{'Time (s)':<15}")
    print("-" * 80)
    print(f"{'AD-PINN':<20}{nu_ad:<25.10e}{rel_err_ad:<25.10e}{time_inverse_ad:<15.4f}")
    print(f"{'FDM-PINN':<20}{nu_fdm:<25.10e}{rel_err_fdm:<25.10e}{time_inverse_fdm:<15.4f}")
    
    print("\nDone! All plots saved.")