PRESETS = {
    # Пресеты
    "Классическое уравнение теплопроводности": {
        "model_name": "Fourier (Classical)",
        "params": {"kappa": 0.01, "tau_R": 0.0, "kappa_tilde": 0.5},
        "domain": {"x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0},
        "ic_expr": "sin(pi*x)", "ict_expr": "0",
        "left_bc_type": "Dirichlet", "right_bc_type": "Dirichlet",
        "left_bc_expr": "0", "right_bc_expr": "0",
        "q_expr": "0",
        "epochs": 2000, "n_f": 5000, "n_b": 200, "n_i": 200, "lr": 1e-3,
        "hidden_layers": 4, "hidden_units": 100
    },
    "Теория Максвелла–Каттанео": {
        "model_name": "Maxwell–Cattaneo",
        "params": {"kappa": 0.01, "tau_R": 0.1, "kappa_tilde": 0.5},
        "domain": {"x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0},
        "ic_expr": "sin(pi*x)", "ict_expr": "0",
        "left_bc_type": "Dirichlet", "right_bc_type": "Dirichlet",
        "left_bc_expr": "0", "right_bc_expr": "0",
        "q_expr": "0",
        "epochs": 2000, "n_f": 7000, "n_b": 200, "n_i": 200, "lr": 1e-3,
        "hidden_layers": 4, "hidden_units": 120
    },
    "Теория Грина–Нагди II-го типа": {
        "model_name": "Green–Naghdi II",
        "params": {"kappa": 0.0, "tau_R": 0.0, "kappa_tilde": 0.5},
        "domain": {"x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0},
        "ic_expr": "sin(pi*x)", "ict_expr": "0",
        "left_bc_type": "Dirichlet", "right_bc_type": "Dirichlet",
        "left_bc_expr": "0", "right_bc_expr": "0",
        "q_expr": "0",
        "epochs": 2500, "n_f": 8000, "n_b": 200, "n_i": 200, "lr": 1e-3,
        "hidden_layers": 5, "hidden_units": 120
    },
    "Теория Грина–Нагди III-го типа": {
        "model_name": "Green–Naghdi III",
        "params": {"kappa": 0.01, "tau_R": 0.0, "kappa_tilde": 0.5},
        "domain": {"x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0},
        "ic_expr": "sin(pi*x)", "ict_expr": "0",
        "left_bc_type": "Dirichlet", "right_bc_type": "Dirichlet",
        "left_bc_expr": "0", "right_bc_expr": "0",
        "q_expr": "0",
        "epochs": 2500, "n_f": 8000, "n_b": 200, "n_i": 200, "lr": 1e-3,
        "hidden_layers": 5, "hidden_units": 120
    },
}
