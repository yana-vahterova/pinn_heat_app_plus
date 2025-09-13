import torch
from .utils import gradients, laplacian_1d

def _as_tensor_like_u(val, u):
    if val is None:
        return torch.zeros_like(u)
    if torch.is_tensor(val):
        return val.to(device=u.device, dtype=u.dtype)
    return torch.full_like(u, float(val))

def residual_classic(model, x, t, params, q_fun=None):
    x.requires_grad_(True); t.requires_grad_(True)
    u = model(x, t)
    ut = gradients(u, t)
    uxx = laplacian_1d(u, x)
    kappa = torch.tensor(params.get("kappa", 1.0), device=x.device, dtype=x.dtype)
    q = _as_tensor_like_u(q_fun(x, t), u) if q_fun is not None else torch.zeros_like(u)
    return ut - kappa * uxx - q

def residual_cattaneo(model, x, t, params, q_fun=None):
    x.requires_grad_(True); t.requires_grad_(True)
    u = model(x, t)
    ut = gradients(u, t)
    utt = gradients(ut, t)
    uxx = laplacian_1d(u, x)
    kappa = torch.tensor(params.get("kappa", 1.0), device=x.device, dtype=x.dtype)
    tauR  = torch.tensor(params.get("tau_R", 0.0), device=x.device, dtype=x.dtype)
    if q_fun is not None:
        q = _as_tensor_like_u(q_fun(x, t), u)
        qt = gradients(q, t)  # если q константа по t -> вернутся нули формы как у t
    else:
        q = torch.zeros_like(u)
        qt = torch.zeros_like(t)
    return ut + tauR * utt - kappa * uxx - (q + tauR * qt)

def residual_gn2(model, x, t, params, q_fun=None):
    x.requires_grad_(True); t.requires_grad_(True)
    u = model(x, t)
    ut = gradients(u, t)
    utt = gradients(ut, t)
    uxx = laplacian_1d(u, x)
    ktilde = torch.tensor(params.get("kappa_tilde", 1.0), device=x.device, dtype=x.dtype)
    if q_fun is not None:
        q = _as_tensor_like_u(q_fun(x, t), u)
        qt = gradients(q, t)
    else:
        qt = torch.zeros_like(t)
    return utt - uxx - qt

def residual_gn3(model, x, t, params, q_fun=None):
    x.requires_grad_(True); t.requires_grad_(True)
    u = model(x, t)
    ut = gradients(u, t)
    utt = gradients(ut, t)
    ux = gradients(u, x)
    uxx = gradients(ux, x)
    utxx = gradients(gradients(ut, x), x)
    kappa = torch.tensor(params.get("kappa", 0.0), device=x.device, dtype=x.dtype)
    ktilde = torch.tensor(params.get("kappa_tilde", 1.0), device=x.device, dtype=x.dtype)
    if q_fun is not None:
        q = _as_tensor_like_u(q_fun(x, t), u)
        qt = gradients(q, t)
    else:
        qt = torch.zeros_like(t)
    return utt - kappa * utxx - uxx - qt

