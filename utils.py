import math
import torch

_ALLOWED_TORCH = {
    "sin": torch.sin,
    "cos": torch.cos,
    "exp": torch.exp,
    "tanh": torch.tanh,
    "sqrt": torch.sqrt,
    "log": torch.log,
    "abs": torch.abs,
    "pow": torch.pow,
    "min": torch.minimum,
    "max": torch.maximum,
}
_ALLOWED_CONST = {"pi": math.pi, "e": math.e}

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def make_callable(expr: str):
    """
    Компилирует строковое выражение в функцию f(x,t) с семантикой torch.
    Если результат — константа/число, приводим к тензору формы как у x.
    """
    expr = (expr or "").strip()
    if expr == "":
        return lambda x, t: torch.zeros_like(x)

    code = compile(expr, "<user-expr>", "eval")

    def f(x, t):
        local_ctx = {"x": x, "t": t}
        local_ctx.update(_ALLOWED_TORCH)
        local_ctx.update(_ALLOWED_CONST)
        res = eval(code, {"__builtins__": {}}, local_ctx)
        if torch.is_tensor(res):
            return res.to(device=x.device, dtype=x.dtype)
        else:
            # константа -> тензор нужной формы/типа/устройства
            return torch.full_like(x, float(res))
    return f

def gradients(y, x, grad_outputs=None, allow_unused=True):
    """
    Безопасная обертка torch.autograd.grad:
    - если y не тензор или не зависит от x -> возвращаем нули формы как у x
    - иначе стандартный градиент; None -> заменяем нулями
    """
    if not (isinstance(y, torch.Tensor) and y.requires_grad):
        return torch.zeros_like(x)
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    g = torch.autograd.grad(
        y, x, grad_outputs=grad_outputs, create_graph=True,
        retain_graph=True, allow_unused=True
    )[0]
    if g is None:
        g = torch.zeros_like(x)
    return g

def laplacian_1d(u, x):
    ux = gradients(u, x)
    uxx = gradients(ux, x)
    return uxx

# =================== Dual-backend make_callable (NumPy/Torch auto) ===================
def make_callable(expr: str):
    """
    Создаёт f(x[, t]) из строки expr. Автовыбор бэкенда:
      • если входы — torch.Tensor ИЛИ в expr есть "torch.", считаем через torch;
      • иначе считаем через NumPy/Math (работает с float).
    Возвращает float для скалярных результатов.
    """
    import math, numpy as np
    try:
        import torch as _th
    except Exception:
        _th = None
    expr = (expr or "").strip()
    code = compile(expr, "<user-expr>", "eval")

    def f(x, t=0.0):
        nonlocal _th
        # Определяем бэкенд
        use_torch = False
        if _th is not None:
            if isinstance(x, getattr(_th, "Tensor", tuple())) or isinstance(t, getattr(_th, "Tensor", tuple())):
                use_torch = True
            elif "torch." in expr:
                use_torch = True

        if use_torch and _th is not None:
            # torch-бэкенд: приведём скаляры к тензорам
            x_th = x if isinstance(x, _th.Tensor) else _th.tensor(x, dtype=_th.float32)
            t_th = t if isinstance(t, _th.Tensor) else _th.tensor(t, dtype=_th.float32)
            local_ctx = {
                "x": x_th, "t": t_th, "torch": _th, "np": np, "numpy": np, "math": math,
                # Частые функции напрямую на torch
                "sin": _th.sin, "cos": _th.cos, "tan": _th.tan,
                "exp": _th.exp, "log": _th.log, "sqrt": _th.sqrt, "abs": _th.abs,
                "pi": math.pi, "e": math.e,
            }
            res = eval(code, {"__builtins__": {}}, local_ctx)
            # Вернём float, если это скалярный тензор
            if isinstance(res, _th.Tensor) and res.numel() == 1:
                return float(res.detach().cpu().item())
            return res
        else:
            # NumPy/Math-бэкенд (все операции должны уметь принять float/ndarray)
            x_np = float(x) if isinstance(x, (int, float)) else x
            t_np = float(t) if isinstance(t, (int, float)) else t
            local_ctx = {
                "x": x_np, "t": t_np, "np": np, "numpy": np, "math": math,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
                "pi": math.pi, "e": math.e,
                # простая ступенчатая
                "heaviside": lambda z: np.heaviside(z, 0.0),
            }
            res = eval(code, {"__builtins__": {}}, local_ctx)
            if isinstance(res, (int, float, np.floating)):
                return float(res)
            return res

    return f

