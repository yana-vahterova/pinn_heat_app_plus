import torch, torch.nn as nn
class PINNModel(nn.Module):
    def __init__(self,in_dim=2,out_dim=1,hidden_layers=4,hidden_units=100,activation='tanh'):
        super().__init__(); acts={'tanh':nn.Tanh(),'relu':nn.ReLU(),'gelu':nn.GELU(),'silu':nn.SiLU()}
        act=acts.get(activation,nn.Tanh()); layers=[]; last=in_dim
        for _ in range(hidden_layers): layers += [nn.Linear(last,hidden_units), act]; last=hidden_units
        layers.append(nn.Linear(last,out_dim)); self.net=nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,x,t): return self.net(torch.cat([x,t],dim=-1))

class TransformedPINN(nn.Module):
    """Hard-enforce zero IC/BC via output transform (used only in zero-only mode)."""
    def __init__(self, base: nn.Module, model_name: str, domain: dict, bc_types: dict):
        super().__init__()
        self.base = base
        self.model_name = model_name
        self.domain = dict(domain)

    def _is_hyperbolic(self) -> bool:
        return self.model_name in ["Maxwell–Cattaneo", "Green–Naghdi II", "Green–Naghdi III"]

    def forward(self, x, t):
        x_min = float(self.domain.get("x_min", 0.0)); x_max = float(self.domain.get("x_max", 1.0))
        t_min = float(self.domain.get("t_min", 0.0)); t_max = float(self.domain.get("t_max", 1.0))
        L = (x_max - x_min) or 1.0
        xi = (x - x_min)/L
        tau = (t - t_min)/((t_max - t_min) + 1e-12)
        # Zero IC
        Tfac = tau**2 if self._is_hyperbolic() else tau
        # Zero BC base (only S*N remains)
        S = (xi**2)*((1.0 - xi)**2)
        return (S * Tfac) * self.base(x,t)


class FlexibleTransformedPINN(nn.Module):
    """
    Per-condition hard enforcement:
      - IC zero: multiply by Tfac (tau or tau^2).
      - Dirichlet zero at side: multiply output by xi or (1-xi) factor.
      - Neumann zero at side: enforce by input transform to make u_x(side)=0:
            left: use xi2 = xi^2 as a feature (chain rule -> u_x ~ xi -> 0 at xi=0)
            right: use xr2 = (1-xi)^2 as a feature (u_x -> 0 at xi=1)
    """
    def __init__(self, base: nn.Module, model_name: str, domain: dict, bc_types: dict,
                 enforce_ic_zero: bool,
                 enforce_ict_zero: bool,
                 enforce_left_zero: bool,
                 enforce_right_zero: bool,
                 left_kind: str = "Dirichlet",
                 right_kind: str = "Dirichlet"):
        super().__init__()
        self.base = base
        self.model_name = str(model_name)
        self.domain = dict(domain)
        self.bc_types = dict(bc_types)
        self.enf_ic  = bool(enforce_ic_zero)
        self.enf_ict = bool(enforce_ict_zero)
        self.enf_L   = bool(enforce_left_zero)
        self.enf_R   = bool(enforce_right_zero)
        self.left_kind  = left_kind
        self.right_kind = right_kind

    def _is_hyperbolic(self) -> bool:
        return self.model_name in ["Maxwell–Cattaneo", "Green–Naghdi II", "Green–Naghdi III"]

    def forward(self, x, t):
        x_min = float(self.domain.get("x_min", 0.0)); x_max = float(self.domain.get("x_max", 1.0))
        t_min = float(self.domain.get("t_min", 0.0)); t_max = float(self.domain.get("t_max", 1.0))
        L = (x_max - x_min) or 1.0
        xi  = (x - x_min)/L
        one_minus = 1.0 - xi
        tau = (t - t_min)/((t_max - t_min) + 1e-12)

        # Time factor: enforce u(x,t_min)=0 and if hyperbolic enforce u_t(x,t_min)=0
        Tfac = 1.0
        if self.enf_ic:
            Tfac = tau if not self._is_hyperbolic() else tau**2

        # Spatial Dirichlet factors
        Sfac = 1.0
        if self.enf_L and self.left_kind == "Dirichlet":
            Sfac = Sfac * xi
        if self.enf_R and self.right_kind == "Dirichlet":
            Sfac = Sfac * one_minus

        # Spatial Neumann-zero via input transform (exact u_x=0 on enforced sides)
        # Build features for base: [phiL, phiR, tau]
        phiL = xi
        phiR = one_minus
        if self.enf_L and self.left_kind == "Neumann":
            phiL = xi**2
        if self.enf_R and self.right_kind == "Neumann":
            phiR = one_minus**2

        # Combine to inputs and evaluate base
        z = torch.cat([phiL, phiR, t], dim=-1)  # keep original time feature for expressiveness
        out = self.base(z[..., :1], z[..., 2:])  # base expects (x,t); feed pseudo-x=phiL, t as original t
        # (we still pass 2D input, but we already concatenated; slicing back to (x_feature,t))

        return (Sfac * Tfac) * out
