import torch
def sample_interior(n_f,x_min,x_max,t_min,t_max,device):
    x=torch.rand(n_f,1,device=device)*(x_max-x_min)+x_min
    t=torch.rand(n_f,1,device=device)*(t_max-t_min)+t_min
    x.requires_grad_(True); t.requires_grad_(True); return x,t
def sample_boundary(n_b,side,x_min,x_max,t_min,t_max,device):
    xv=x_min if side=='left' else x_max
    x=torch.full((n_b,1),float(xv),device=device)
    t=torch.rand(n_b,1,device=device)*(t_max-t_min)+t_min
    x.requires_grad_(True); t.requires_grad_(True); return x,t
def sample_initial(n_i,x_min,x_max,t0,device):
    x=torch.rand(n_i,1,device=device)*(x_max-x_min)+x_min
    t=torch.full((n_i,1),float(t0),device=device)
    x.requires_grad_(True); t.requires_grad_(True); return x,t
