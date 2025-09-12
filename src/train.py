import os
import torch
import torch.optim as optim

from .utils import gradients
from .pdes import residual_classic, residual_cattaneo, residual_gn2, residual_gn3
from .data import sample_interior, sample_boundary, sample_initial

def make_residual_fn(model_name):
    if model_name == "Fourier (Classical)": return residual_classic
    if model_name == "Maxwell–Cattaneo":   return residual_cattaneo
    if model_name == "Green–Naghdi II":    return residual_gn2
    if model_name == "Green–Naghdi III":   return residual_gn3
    raise ValueError(f"Unknown model: {model_name}")

def _maybe_save_checkpoint(model, info, loss_val):
    ck = info["ckpt"]
    if not ck["enable"]:
        return
    os.makedirs(ck["dir"], exist_ok=True)
    if ck["mode"] == "best":
        if loss_val < info["best_loss"]:
            info["best_loss"] = float(loss_val)
            info["last_path"] = os.path.join(ck["dir"], "best.pth")
            torch.save({"state_dict": model.state_dict(), "best_loss": info["best_loss"]}, info["last_path"])
            print(f"[CKPT] Saved best: {info['last_path']} (loss={info['best_loss']:.6e})")
    else:
        # periodic
        ep = info["epoch"]
        if ck["every"] > 0 and (ep % ck["every"] == 0):
            path = os.path.join(ck["dir"], f"epoch_{ep}.pth")
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "loss": float(loss_val)}, path)
            info["last_path"] = path
            print(f"[CKPT] Saved periodic: {path} (loss={float(loss_val):.6e})")


def _pick_device(device_str: str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps":
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")

def train_PINN(model, config, q_fun=None, ic_fun=None, ict_fun=None, bc_left=None, bc_right=None, device="cpu"):
    dev = _pick_device(device)
    dev = _pick_device(device)
    model = model.to(device).train()
    model_name = config["model_name"]
    params     = config["params"]
    hard = config.get("hard_enforce", {})
    domain     = config["domain"]

    n_f    = int(config.get("n_f", 10000))
    n_b    = int(config.get("n_b", 200))
    n_i    = int(config.get("n_i", 200))
    epochs = int(config.get("epochs", 2000))
    lr     = float(config.get("lr", 1e-3))
    bc_types = config.get("bc_types", {"left":"Dirichlet","right":"Dirichlet"})

    w = config.get("loss_weights", {"pde":1.0,"ic":1.0,"bc":1.0})
    wpde, wic, wbc = float(w.get("pde",1.0)), float(w.get("ic",1.0)), float(w.get("bc",1.0))

    ckpt_cfg = config.get("checkpoint", {"enable":False,"dir":"checkpoints","mode":"best","every":500})

    x_min = float(domain.get("x_min", 0.0)); x_max = float(domain.get("x_max", 1.0))
    t_min = float(domain.get("t_min", 0.0)); t_max = float(domain.get("t_max", 1.0))

    residual_fn = make_residual_fn(model_name)
    optimizer   = optim.Adam(model.parameters(), lr=lr)
    history = []

    is_hyperbolic = (model_name in ["Maxwell–Cattaneo", "Green–Naghdi II", "Green–Naghdi III"])

    info = {"best_loss": float("inf"), "last_path": None, "ckpt": ckpt_cfg, "epoch": 0}

    for ep in range(1, epochs+1):
        info["epoch"] = ep
        optimizer.zero_grad()

        # PDE residual
        xf, tf = sample_interior(n_f, x_min, x_max, t_min, t_max, device)
        res = residual_fn(model, xf, tf, params, q_fun=q_fun)
        loss_pde = torch.mean(res**2)

        # Initial conditions
        xi, ti = sample_initial(n_i, x_min, x_max, t_min, device)
        loss_ic = torch.tensor(0.0, device=device)
        # hard-enforce: skip corresponding terms
        if ic_fun is not None and not bool(hard.get('ic_zero', False)):
            u0_pred = model(xi, ti); u0_true = ic_fun(xi, ti)
            loss_ic = loss_ic + torch.mean((u0_pred - u0_true)**2)
        if is_hyperbolic and ict_fun is not None and not bool(hard.get('ict_zero', False)):
            ut_pred = gradients(model(xi, ti), ti); ut_true = ict_fun(xi, ti)
            loss_ic = loss_ic + torch.mean((ut_pred - ut_true)**2)

        # Boundary conditions
        loss_bc = torch.tensor(0.0, device=device)
        # left
        if not bool(hard.get('left_zero', False)):
            xbL, tbL = sample_boundary(n_b, "left", x_min, x_max, t_min, t_max, device)
            if bc_types.get("left","Dirichlet") == "Dirichlet":
                uL = model(xbL, tbL); gL = bc_left(xbL, tbL) if bc_left is not None else torch.zeros_like(uL)
                loss_bc = loss_bc + torch.mean((uL - gL)**2)
            else:
                uL = model(xbL, tbL); uxL = gradients(uL, xbL)
                gL = bc_left(xbL, tbL) if bc_left is not None else torch.zeros_like(uL)
                loss_bc = loss_bc + torch.mean((uxL - gL)**2)
        # right
        if not bool(hard.get('right_zero', False)):
            xbR, tbR = sample_boundary(n_b, "right", x_min, x_max, t_min, t_max, device)
            if bc_types.get("right","Dirichlet") == "Dirichlet":
                uR = model(xbR, tbR); gR = bc_right(xbR, tbR) if bc_right is not None else torch.zeros_like(uR)
                loss_bc = loss_bc + torch.mean((uR - gR)**2)
            else:
                uR = model(xbR, tbR); uxR = gradients(uR, xbR)
                gR = bc_right(xbR, tbR) if bc_right is not None else torch.zeros_like(uR)
                loss_bc = loss_bc + torch.mean((uxR - gR)**2)

        # Weighted sum
        loss = wpde*loss_pde + wic*loss_ic + wbc*loss_bc
        loss.backward(); optimizer.step()
        L = float(loss.detach().cpu().item())
        history.append(L)

        # checkpoints
        _maybe_save_checkpoint(model, info, L)

        if ep % max(1, epochs//10) == 0 or ep == 1:
            print(f"Epoch {ep:5d}/{epochs}: loss={L:.6e} | pde={loss_pde.item():.3e} ic={loss_ic.item():.3e} bc={loss_bc.item():.3e}")

    return model, history, info



