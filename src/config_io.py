import json

UI_KEYS = [
    "model_name","kappa","tau_R","kappa_tilde","x_min","x_max","t_min","t_max",
    "epochs","n_f","n_b","n_i","lr","hidden_layers","hidden_units",
    "left_bc_type","right_bc_type","ic_expr","ict_expr","left_bc_expr","right_bc_expr","q_expr",
    # Новые поля UI:
    "w_pde","w_ic","w_bc",
    "save_ckpt","ckpt_dir","ckpt_mode","ckpt_every",
    "use_output_transform_zero_only"
]

def make_export_dict(state):
    d = {}
    for k in UI_KEYS:
        d[k] = state.get(k, None)
    d["bc_types"] = {"left": d.get("left_bc_type","Dirichlet"), "right": d.get("right_bc_type","Dirichlet")}
    return d

def apply_config_to_state(state, cfg: dict):
    for k in UI_KEYS:
        if k in cfg:
            state[k] = cfg[k]
    if "params" in cfg:
        p = cfg["params"]
        if "kappa" in p: state["kappa"] = p["kappa"]
        if "tau_R" in p: state["tau_R"] = p["tau_R"]
        if "kappa_tilde" in p: state["kappa_tilde"] = p["kappa_tilde"]
    if "domain" in cfg:
        d = cfg["domain"]
        for z in ["x_min","x_max","t_min","t_max"]:
            if z in d: state[z] = d[z]
    if "bc_types" in cfg:
        b = cfg["bc_types"]
        if "left" in b:  state["left_bc_type"]  = b["left"]
        if "right" in b: state["right_bc_type"] = b["right"]

def build_train_config(state):
    cfg = {
        "model_name": state["model_name"],
        "params": {
            "kappa": float(state["kappa"]),
            "tau_R": float(state["tau_R"]),
            "kappa_tilde": float(state["kappa_tilde"])
        },
        "domain": {
            "x_min": float(state["x_min"]),
            "x_max": float(state["x_max"]),
            "t_min": float(state["t_min"]),
            "t_max": float(state["t_max"])
        },
        "n_f": int(state["n_f"]),
        "n_b": int(state["n_b"]),
        "n_i": int(state["n_i"]),
        "epochs": int(state["epochs"]),
        "lr": float(state["lr"]),
        "bc_types": {"left": state["left_bc_type"], "right": state["right_bc_type"]},
        # Новое: веса лосса
        "loss_weights": {
            "pde": float(state.get("w_pde", 1.0)),
            "ic":  float(state.get("w_ic",  1.0)),
            "bc":  float(state.get("w_bc",  1.0))
        },
        # Новое: чекпоинты
        "checkpoint": {
            "enable": bool(state.get("save_ckpt", False)),
            "dir": state.get("ckpt_dir", "checkpoints"),
            "mode": state.get("ckpt_mode", "best"),   # best | periodic
            "every": int(state.get("ckpt_every", 500))
        }
    }
    return cfg
