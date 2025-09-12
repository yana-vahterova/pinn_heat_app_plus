# ==== DEVICE SELECTION BLOCK BEGIN ====

import json
import math
import streamlit as st
import torch

# ==== DEVICE SELECTION BLOCK BEGIN ====
def _available_devices():
    opts = ["cpu"]
    try:
        if torch.cuda.is_available():
            opts.append("cuda")
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            opts.append("mps")
    except Exception:
        pass
    return opts

if "device" not in st.session_state:
    st.session_state["device"] = "cuda" if torch.cuda.is_available() else "cpu"

with st.sidebar.expander("Устройство (для обучения PINN)", expanded=True):
    dev_opts = _available_devices()
    default_idx = dev_opts.index(st.session_state.get("device", dev_opts[0])) if st.session_state.get("device", dev_opts[0]) in dev_opts else 0
    choice = st.radio("Выберите устройство", dev_opts, index=default_idx, key="device")
    if choice == "cuda" and torch.cuda.is_available():
        try:
            st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            st.caption("GPU: CUDA")
    elif choice == "mps":
        st.caption("Apple Metal (MPS)")
    else:
        st.caption("CPU")

_selected_device = st.session_state["device"]
if _selected_device == "cuda" and not torch.cuda.is_available():
    st.warning("CUDA недоступна — переключаюсь на CPU.")
    _selected_device = "cpu"
if _selected_device == "mps":
    try:
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            st.warning("MPS недоступна — переключаюсь на CPU.")
            _selected_device = "cpu"
    except Exception:
        st.warning("MPS не поддерживается — переключаюсь на CPU.")
        _selected_device = "cpu"
# ==== DEVICE SELECTION BLOCK END ====

import torch
import pandas as pd
import plotly.graph_objects as go

from src.utils import get_device, make_callable
from src.pinn_model import PINNModel, TransformedPINN, FlexibleTransformedPINN
from src.train import train_PINN
from src.presets import PRESETS
from src.config_io import make_export_dict, apply_config_to_state, build_train_config
from src.numerics import solve_numeric
from src.help_tab import render_help

st.set_page_config(page_title="PINN Heat Conduction", layout="wide")
st.title("🔬 РЕШЕНИЕ УРАВНЕНИЙ КЛАССИЧЕСКОЙ И ОБОБЩЕННОЙ МОДЕЛЕЙ ТЕПЛОПРОВОДНОСТИ С ПОМОЩЬЮ PINN (PyTorch + Streamlit)")


def _is_zero_expr(s: str) -> bool:
    if s is None: return True
    s = str(s).strip().lower()
    return s in ("", "0", "0.0", "0.", "sin(0)", "cos(pi/2)")  # quick conservative check

def ss_set_default(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

# Значения по умолчанию
ss_defaults = {
    "model_name": "Fourier (Classical)",
    "kappa": 0.01, "tau_R": 0.0, "kappa_tilde": 0.5,
    "x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0,
    "epochs": 2000, "n_f": 5000, "n_b": 200, "n_i": 200, "lr": 1e-3,
    "hidden_layers": 4, "hidden_units": 100,
    "left_bc_type": "Dirichlet", "right_bc_type": "Dirichlet",
    "ic_expr": "sin(pi*x)", "ict_expr": "0",
    "left_bc_expr": "0", "right_bc_expr": "0",
    "q_expr": "0",
    # Веса лосса и чекпоинты
    "w_pde": 1.0, "w_ic": 1.0, "w_bc": 1.0,
    "save_ckpt": True, "ckpt_dir": "checkpoints", "ckpt_mode": "best", "ckpt_every": 500,
    # Флаг и место для последней обученной модели
    "has_model": False, "use_output_transform_zero_only": False
}
for k, v in ss_defaults.items():
    ss_set_default(k, v)

# --- Сайдбар: пресеты и конфиги ---
with st.sidebar:
    st.header("Пресеты и конфиги")
    
    st.markdown("---")
    st.checkbox("Hard‑enforce zero IC/BC when they are zero", key="use_output_transform_zero_only",
                help="Если начальные/граничные условия равны нулю, жёстко обнуляет их через output transform; иначе условия учитываются в функции потерь.")
    preset_name = st.selectbox("Выбрать пресет", ["", *PRESETS.keys()])
    colp1, colp2 = st.columns(2)
    if colp1.button("Загрузить пресет в форму", use_container_width=True):
        if preset_name and preset_name in PRESETS:
            apply_config_to_state(st.session_state, PRESETS[preset_name])
            st.success(f"Загружен пресет: {preset_name}")
            st.rerun()
    if colp2.button("Сбросить форму", use_container_width=True):
        for k, v in ss_defaults.items():
            st.session_state[k] = v
        st.rerun()

    st.markdown("— **Сохранить конфиг**")
    export = make_export_dict(st.session_state)
    st.download_button(
        "💾 Скачать JSON",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name="pinn_config.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("— **Загрузить конфиг**")
    up = st.file_uploader("JSON-файл конфига", type=["json"], label_visibility="collapsed")
    if up is not None:
        try:
            cfg = json.load(up)
            apply_config_to_state(st.session_state, cfg)
            st.success("Конфиг загружен")
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")

# --- Сайдбар: постановка задачи ---
with st.sidebar:
    st.header("Постановка задачи")
    model_list = ["Fourier (Classical)", "Maxwell–Cattaneo", "Green–Naghdi II", "Green–Naghdi III"]
    st.session_state["model_name"] = st.selectbox(
        "Модель теплопроводности",
        model_list,
        index=model_list.index(st.session_state["model_name"])
    )

    st.markdown("**Коэффициенты**")
    _model = st.session_state["model_name"]
    # Показать только релевантные коэффициенты и обнулить лишние
    if _model == "Fourier (Classical)":
        st.session_state["kappa"] = st.number_input("kappa (κ)", value=float(st.session_state["kappa"]), format="%.6f")
        st.session_state["tau_R"] = 0.0
        st.session_state["kappa_tilde"] = 0.0
        st.caption("Для модели Фурье параметры τ_R и ~κ не используются и установлены в 0.")
    elif _model == "Maxwell–Cattaneo":
        st.session_state["kappa"] = st.number_input("kappa (κ)", value=float(st.session_state["kappa"]), format="%.6f")
        st.session_state["tau_R"] = st.number_input("tau_R (τ_R)", value=float(st.session_state["tau_R"]), format="%.6f")
        st.session_state["kappa_tilde"] = 0.0
        st.caption("Для Каттанео параметр ~κ не используется и установлен в 0.")
    elif _model == "Green–Naghdi II":
        st.session_state["kappa"] = 0.0
        st.session_state["tau_R"] = 0.0
        st.session_state["kappa_tilde"] = st.number_input("kappa_tilde (~κ)", value=float(st.session_state["kappa_tilde"]), format="%.6f")
        st.caption("Для GN-II параметры κ и τ_R не используются и установлены в 0.")
    elif _model == "Green–Naghdi III":
        st.session_state["kappa"] = st.number_input("kappa (κ)", value=float(st.session_state["kappa"]), format="%.6f")
        st.session_state["tau_R"] = 0.0
        st.session_state["kappa_tilde"] = st.number_input("kappa_tilde (~κ)", value=float(st.session_state["kappa_tilde"]), format="%.6f")
        st.caption("Для GN-III параметр τ_R не используется и установлен в 0.")

    st.markdown("**Область**")
    st.session_state["x_min"] = st.number_input("x_min", value=float(st.session_state["x_min"]), format="%.3f")
    st.session_state["x_max"] = st.number_input("x_max", value=float(st.session_state["x_max"]), format="%.3f")
    st.session_state["t_min"] = st.number_input("t_min", value=float(st.session_state["t_min"]), format="%.3f")
    st.session_state["t_max"] = st.number_input("t_max", value=float(st.session_state["t_max"]), format="%.3f")

    st.markdown("**PINN**")
    st.session_state["epochs"] = st.number_input("epochs", value=int(st.session_state["epochs"]), step=100)
    st.session_state["n_f"] = st.number_input("N_interior (коллокац.)", value=int(st.session_state["n_f"]), step=500)
    st.session_state["n_b"] = st.number_input("N_boundary (на границу)", value=int(st.session_state["n_b"]), step=50)
    st.session_state["n_i"] = st.number_input("N_initial", value=int(st.session_state["n_i"]), step=50)
    st.session_state["lr"] = st.number_input("learning rate", value=float(st.session_state["lr"]), format="%.5f")

    st.markdown("**Архитектура сети**")
    st.session_state["hidden_layers"] = st.number_input("hidden layers", value=int(st.session_state["hidden_layers"]))
    st.session_state["hidden_units"] = st.number_input("hidden units", value=int(st.session_state["hidden_units"]))

    st.divider()
    st.markdown("**Веса лосса**")
    st.session_state["w_pde"] = st.number_input("w_pde (PDE)", value=float(st.session_state["w_pde"]), format="%.3f")
    st.session_state["w_ic"]  = st.number_input("w_ic (IC)",  value=float(st.session_state["w_ic"]),  format="%.3f")
    st.session_state["w_bc"]  = st.number_input("w_bc (BC)",  value=float(st.session_state["w_bc"]),  format="%.3f")

    st.markdown("**Чекпоинты**")
    st.session_state["save_ckpt"] = st.checkbox("Сохранять чекпоинты", value=bool(st.session_state["save_ckpt"]))
    if st.session_state["save_ckpt"]:
        st.session_state["ckpt_dir"] = st.text_input("Папка для чекпоинтов", value=str(st.session_state["ckpt_dir"]))
        st.session_state["ckpt_mode"] = st.radio("Режим", ["best", "periodic"], index=["best","periodic"].index(st.session_state["ckpt_mode"]))
        if st.session_state["ckpt_mode"] == "periodic":
            st.session_state["ckpt_every"] = st.number_input("Сохранять каждые N эпох", min_value=1, value=int(st.session_state["ckpt_every"]), step=50)

# --- Центр: IC/BC и источник ---
st.subheader("Начальные и граничные условия, источник")
# --- Выбор типа граничных условий ---
bc_col1, bc_col2, bc_col3 = st.columns([1,1,2])
with bc_col1:
    st.session_state["left_bc_type"] = st.selectbox(
        "Левая граница @ x=x_min",
        options=["Dirichlet", "Neumann"],
        index=["Dirichlet","Neumann"].index(st.session_state["left_bc_type"])
    )
with bc_col2:
    st.session_state["right_bc_type"] = st.selectbox(
        "Правая граница @ x=x_max",
        options=["Dirichlet", "Neumann"],
        index=["Dirichlet","Neumann"].index(st.session_state["right_bc_type"])
    )
with bc_col3:
    if st.session_state["left_bc_type"] == "Dirichlet" and st.session_state["right_bc_type"] == "Dirichlet":
        st.caption("Dirichlet: задаётся значение T.  Neumann: задаётся поток ∂T/∂x (с направлением +x).")
    else:
        st.caption("Neumann означает задать значение производной ∂T/∂x на границе (тепловой поток). Dirichlet — значение T.")


col1, col2 = st.columns(2)
with col1:
    st.session_state["ic_expr"]  = st.text_area("IC: T(x, 0) =", value=st.session_state["ic_expr"])
    st.session_state["ict_expr"] = st.text_area("IC (для 2-го порядка): T_t(x, 0) =", value=st.session_state["ict_expr"])
with col2:
    left_label  = f"{st.session_state['left_bc_type']} @ x=x_min"
    right_label = f"{st.session_state['right_bc_type']} @ x=x_max"
    st.session_state["left_bc_expr"]  = st.text_area(left_label,  value=st.session_state["left_bc_expr"])
    st.session_state["right_bc_expr"] = st.text_area(right_label, value=st.session_state["right_bc_expr"])

st.session_state["q_expr"] = st.text_area("Источник q(x,t) =", value=st.session_state["q_expr"])
st.caption("Доступны функции torch: sin, cos, exp, tanh, sqrt, log, abs, pow, min, max; константы pi, e; переменные x, t.")

# --- Кнопка запуска обучения ---
run = st.button("🚀 Запустить обучение")
device = get_device()
st.write(f"**Устройство:** {device}")

# --- Обучение ---
if run:
    # Компилируем выражения
    q_fun        = make_callable(st.session_state["q_expr"])        if st.session_state["q_expr"].strip()        != "" else None
    ic_fun       = make_callable(st.session_state["ic_expr"])       if st.session_state["ic_expr"].strip()       != "" else None
    ict_fun      = make_callable(st.session_state["ict_expr"])      if st.session_state["ict_expr"].strip()      != "" else None
    bc_left_fun  = make_callable(st.session_state["left_bc_expr"])  if st.session_state["left_bc_expr"].strip()  != "" else None
    bc_right_fun = make_callable(st.session_state["right_bc_expr"]) if st.session_state["right_bc_expr"].strip() != "" else None

    
    # Per-condition hardening:
    left_kind  = st.session_state["left_bc_type"]
    right_kind = st.session_state["right_bc_type"]
    ic_zero  = _is_zero_expr(st.session_state.get("ic_expr",""))
    ict_zero = _is_zero_expr(st.session_state.get("ict_expr",""))
    bcl_zero = _is_zero_expr(st.session_state.get("left_bc_expr",""))
    bcr_zero = _is_zero_expr(st.session_state.get("right_bc_expr",""))

    # Only enable if user checked the box
    hard_flags = {
        "ic_zero":  bool(st.session_state.get("use_output_transform_zero_only", False) and ic_zero),
        "ict_zero": bool(st.session_state.get("use_output_transform_zero_only", False) and ict_zero),
        "left_zero":  bool(st.session_state.get("use_output_transform_zero_only", False) and bcl_zero),
        "right_zero": bool(st.session_state.get("use_output_transform_zero_only", False) and bcr_zero),
        "left_kind": left_kind,
        "right_kind": right_kind,
    }
    
    _zero_ic   = _is_zero_expr(st.session_state.get("ic_expr",""))
    _zero_ict  = _is_zero_expr(st.session_state.get("ict_expr",""))
    _zero_bcl  = _is_zero_expr(st.session_state.get("left_bc_expr",""))
    _zero_bcr  = _is_zero_expr(st.session_state.get("right_bc_expr",""))
    _is_hyperb = st.session_state["model_name"] in ["Maxwell–Cattaneo","Green–Naghdi II","Green–Naghdi III"]
    _all_zero  = (_zero_bcl and _zero_bcr and _zero_ic and (not _is_hyperb or _zero_ict))

    # If user enabled the checkbox and all are zero -> turn on hard-enforcement; else ensure it's off
    if st.session_state.get("use_output_transform_zero_only", False) and _all_zero:
        _hard_zero_mode = True
    else:
        _hard_zero_mode = False
    
    train_cfg = build_train_config(st.session_state)
    train_cfg['hard_enforce'] = hard_flags
    # Per-condition skipping happens in train(); keep weights as configured.
    pass

    # Проверки
    def _validate_callable(fn, name):
        if fn is None:
            return True, None
        try:
            xt = torch.tensor([[ (float(st.session_state["x_min"])+float(st.session_state["x_max"])) * 0.5 ]], device=device)
            tt = torch.tensor([[ float(st.session_state["t_min"]) ]], device=device)
            _ = fn(xt, tt)
            return True, None
        except Exception as e:
            return False, f"{name}: {e}"

    problems = []
    for fn, nm in [(ic_fun,"IC T(x,0)"), (ict_fun,"IC_t T_t(x,0)"),
                   (bc_left_fun,"BC left"), (bc_right_fun,"BC right"), (q_fun,"q(x,t)")]:
        ok, err = _validate_callable(fn, nm)
        if not ok: problems.append(err)

    hyper = st.session_state["model_name"] in ["Maxwell–Cattaneo","Green–Naghdi II","Green–Naghdi III"]
    if hyper and ict_fun is None:
        st.info("Для выбранной модели требуется T_t(x,0). Не задано — будет использовано T_t(x,0)=0.")

    for key in ["epochs","n_f","n_b","n_i"]:
        if int(st.session_state[key]) <= 0:
            problems.append(f"{key} должно быть > 0")
    if problems:
        st.error("Найдены проблемы в вводе:")
        for p in problems: st.write(f"• {p}")
        st.stop()

    # Модель и обучение
    model = PINNModel(in_dim=2, out_dim=1,
                      hidden_layers=int(st.session_state["hidden_layers"]),
                      hidden_units=int(st.session_state["hidden_units"]))
    if any([hard_flags['ic_zero'], hard_flags['ict_zero'], hard_flags['left_zero'], hard_flags['right_zero']]):
        domain_cfg = {"x_min": float(st.session_state["x_min"]), "x_max": float(st.session_state["x_max"]),
                      "t_min": float(st.session_state["t_min"]), "t_max": float(st.session_state["t_max"])}
        bc_types_cfg = {"left": st.session_state["left_bc_type"], "right": st.session_state["right_bc_type"]}
        model = FlexibleTransformedPINN(model, st.session_state["model_name"], domain_cfg, bc_types_cfg,
                                      enforce_ic_zero=hard_flags['ic_zero'],
                                      enforce_ict_zero=hard_flags['ict_zero'],
                                      enforce_left_zero=hard_flags['left_zero'],
                                      enforce_right_zero=hard_flags['right_zero'],
                                      left_kind=left_kind, right_kind=right_kind)


    def ic_call(x, t):    return ic_fun(x, t) if ic_fun else torch.zeros_like(x)
    def ict_call(x, t):   return ict_fun(x, t) if ict_fun else torch.zeros_like(x)
    def left_call(x, t):  return bc_left_fun(x, t) if bc_left_fun else torch.zeros_like(x)
    def right_call(x, t): return bc_right_fun(x, t) if bc_right_fun else torch.zeros_like(x)
    q_call = (lambda x, t: q_fun(x, t)) if q_fun else None

    try:
        with st.spinner("Обучение запущено…"):
            model, history, info = train_PINN(model, train_cfg, q_fun=q_call,
                ic_fun=ic_call, ict_fun=ict_call,
                bc_left=left_call, bc_right=right_call
            , device=_selected_device)
        st.success("Обучение завершено ✅")
        if train_cfg["checkpoint"]["enable"]:
            if info.get("last_path"):
                st.info(f"Последний чекпоинт: **{info['last_path']}**.")
            else:
                st.info("Чекпоинтирование включено, но сохранений в ходе обучения не было.")

        # ВАЖНО: сохраняем обученную модель в session_state (на CPU), чтобы ползунки не «обнуляли»
        model_cpu = model.to("cpu").eval()
        st.session_state["trained_model"] = model_cpu
        st.session_state["has_model"] = True

    except Exception as e:
        st.error("Во время обучения возникла ошибка:")
        st.exception(e)
        st.stop()

# --- Универсальная визуализация последней обученной модели ---
st.subheader("Визуализация результата (последняя обученная модель)")
if not st.session_state.get("has_model", False):
    st.info("Пока нет обученной модели. Нажмите «🚀 Запустить обучение».")
else:
    # берём модель из session_state и переносим на текущее устройство
    model = st.session_state["trained_model"].to(get_device()).eval()

    nx = st.slider("nx (сетка по x)", min_value=51, max_value=401, value=201, step=10)
    nt = st.slider("nt (сетка по t)", min_value=51, max_value=401, value=201, step=10)

    xs = torch.linspace(float(st.session_state["x_min"]), float(st.session_state["x_max"]), steps=nx, device=get_device()).view(-1,1)
    ts = torch.linspace(float(st.session_state["t_min"]), float(st.session_state["t_max"]), steps=nt, device=get_device()).view(-1,1)
    X = xs.repeat(1, nt).reshape(-1,1)
    Tt = ts.t().repeat(nx, 1).reshape(-1,1)

    with torch.no_grad():
        U = model(X, Tt)

    U2d = U.reshape(xs.shape[0], ts.shape[0]).detach().cpu().numpy()
    Xv = xs.detach().cpu().numpy().ravel()
    Tv = ts.detach().cpu().numpy().ravel()

    fig = go.Figure(data=[go.Surface(x=Tv, y=Xv, z=U2d, colorbar=dict(title="T"),
                                     contours={"z": {"show": True, "usecolormap": True}})])
    fig.update_layout(scene=dict(xaxis_title="t", yaxis_title="x", zaxis_title="T"), height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Профиль T(x) при выбранном t-индексе
    t_idx = st.slider("Профиль T(x) при t-индексе", min_value=0, max_value=ts.shape[0]-1, value=0)
    t_sel = float(Tv[t_idx])
    st.caption(f"Выбран t ≈ {t_sel:.4f}")
    st.line_chart(pd.DataFrame({"x": Xv, "T(x, t_sel)": U2d[:, t_idx]}).set_index("x"))

    # Профиль T(t) при фиксированном x-индексе
    x_idx = st.slider("Профиль T(t) при x-индексе", min_value=0, max_value=xs.shape[0]-1, value=int(xs.shape[0]//2))
    x_sel = float(Xv[x_idx])
    st.caption(f"Выбран x ≈ {x_sel:.4f}")
    st.line_chart(pd.DataFrame({"t": Tv, "T(t, x_sel)": U2d[x_idx, :]}).set_index("t"))
    # ===== Численное сравнение (FDM) =====
    st.subheader("Численное сравнение (FDM)")

# Метод (с уникальным key) и метрика ошибки
method_label = st.selectbox(
    "Численный метод (FDM)",
    ["Crank–Nicolson (неявная, 2-й порядок)", "Backward Euler (неявная, 1-й порядок)", "FTCS (явная, быстрая)"],
    index=0, key="fdm_method_select"
)
_map = {
    "FTCS (явная, быстрая)": "ftcs",
    "Backward Euler (неявная, 1-й порядок)": "be",
    "Crank–Nicolson (неявная, 2-й порядок)": "cn"
}
fdm_method = _map[method_label]

err_label = st.selectbox(
    "Метрика ошибки",
    ["MSE (абс.)", "RMSE (абс.)", "MAE (абс.)", "RelMSE (%)", "RelMAE (%)", "MaxRel (%)"],
    index=0, key="fdm_error_metric_select"
)

def _err_scalar(pred, true, kind):
    import numpy as np
    pred = np.asarray(pred); true = np.asarray(true)
    diff = pred - true
    if kind == "MSE (абс.)":  return float(np.mean(diff**2))
    if kind == "RMSE (абс.)": return float(np.sqrt(np.mean(diff**2)))
    if kind == "MAE (абс.)":  return float(np.mean(np.abs(diff)))
    if kind == "RelMSE (%)":
        denom = float(np.mean(true**2)) + 1e-12
        return float(100.0*np.mean(diff**2)/denom)
    if kind == "RelMAE (%)":
        denom = float(np.mean(np.abs(true))) + 1e-12
        return float(100.0*np.mean(np.abs(diff))/denom)
    if kind == "MaxRel (%)":
        denom = np.maximum(np.abs(true), 1e-12)
        return float(100.0*np.max(np.abs(diff)/denom))
    return float(np.mean(diff**2))

# Кнопка пересчёта численного решения
if st.button("Построить численное решение (FDM) и сравнить с PINN", key="btn_fdm_compare"):
    q_fun_num        = make_callable(st.session_state["q_expr"])        if st.session_state["q_expr"].strip()        != "" else None
    ic_fun_num       = make_callable(st.session_state["ic_expr"])       if st.session_state["ic_expr"].strip()       != "" else None
    ict_fun_num      = make_callable(st.session_state["ict_expr"])      if st.session_state["ict_expr"].strip()      != "" else None
    bc_left_fun_num  = make_callable(st.session_state["left_bc_expr"])  if st.session_state["left_bc_expr"].strip()  != "" else None
    bc_right_fun_num = make_callable(st.session_state["right_bc_expr"]) if st.session_state["right_bc_expr"].strip() != "" else None

    cfg = {
        "model_name": st.session_state["model_name"],
        "params": {"kappa": float(st.session_state["kappa"]),
                   "tau_R": float(st.session_state["tau_R"]),
                   "kappa_tilde": float(st.session_state["kappa_tilde"])},
        "domain": {"x_min": float(st.session_state["x_min"]),
                   "x_max": float(st.session_state["x_max"]),
                   "t_min": float(st.session_state["t_min"]),
                   "t_max": float(st.session_state["t_max"])},
        "bc_types": {"left": st.session_state["left_bc_type"], "right": st.session_state["right_bc_type"]},
    }

    with st.spinner("Считаем численное решение (FDM)…"):
        U_num, info_num = solve_numeric(
            cfg["model_name"], cfg["params"], cfg["domain"], cfg["bc_types"],
            ic_fun_num, ict_fun_num, bc_left_fun_num, bc_right_fun_num, q_fun_num,
            nx, nt, device="cpu", method=fdm_method
        )

    st.session_state["fdm"] = {"U_num": U_num, "info": info_num, "method": fdm_method, "nx": int(nx), "nt": int(nt)}
    st.success("Численное решение посчитано и сохранено.")

# Визуализация FDM и сравнение
if "fdm" in st.session_state:
    import numpy as np
    U_num = np.array(st.session_state["fdm"]["U_num"])
    info_num = st.session_state["fdm"]["info"]
    if info_num.get("stable_hint"): st.warning(info_num["stable_hint"])
    if U_num.shape != (len(Xv), len(Tv)):
        st.warning("Размеры текущих сеток отличаются от FDM — пересчитайте FDM под текущие nx/nt.")

    figN = go.Figure(data=[go.Surface(x=Tv, y=Xv, z=U_num, colorbar=dict(title="T_num"),
                                      contours={"z":{"show":True,"usecolormap":True}})])
    figN.update_layout(scene=dict(xaxis_title="t", yaxis_title="x", zaxis_title="T_num"), height=600, title="Численное решение (FDM)")
    st.plotly_chart(figN, use_container_width=True)

    if st.session_state.get("has_model", False) and U_num.shape == (len(Xv), len(Tv)):
        err_scalar = _err_scalar(U2d, U_num, err_label)
        st.metric(f"Ошибка PINN vs FDM ({err_label})", f"{err_scalar:.3e}")

    # Срез T(x) при собственном t (с уникальным key)
    t_idx_fdm = st.slider("Срез: T(x) при t-индексе (FDM-сравнение)",
                          min_value=0, max_value=len(Tv)-1, value=min(len(Tv)-1, 0), key="fdm_slice_t_idx")
    t_sel = float(Tv[t_idx_fdm]); st.caption(f"Выбран t ≈ {t_sel:.4f}")
    figC = go.Figure()
    if st.session_state.get("has_model", False) and U_num.shape == (len(Xv), len(Tv)):
        figC.add_trace(go.Scatter(x=Xv, y=U2d[:, t_idx_fdm], mode="lines", name="PINN"))
    figC.add_trace(go.Scatter(x=Xv, y=U_num[:, t_idx_fdm], mode="lines", name="FDM", line=dict(dash="dash")))
    figC.update_layout(xaxis_title="x", yaxis_title="T(x)", height=400, title=f"Срез по x при t≈{t_sel:.4f}")
    st.plotly_chart(figC, use_container_width=True)
    if st.session_state.get("has_model", False) and U_num.shape == (len(Xv), len(Tv)):
        err_slice_x = _err_scalar(U2d[:, t_idx_fdm], U_num[:, t_idx_fdm], err_label)
        st.metric(f"Ошибка на срезе T(x) при t≈{t_sel:.4f} ({err_label})", f"{err_slice_x:.3e}")

    # Срез T(t) при собственном x (с уникальным key)
    x_idx_fdm = st.slider("Срез: T(t) при x-индексе (FDM-сравнение)",
                          min_value=0, max_value=len(Xv)-1, value=int(len(Xv)//2), key="fdm_slice_x_idx")
    x_sel = float(Xv[x_idx_fdm]); st.caption(f"Выбран x ≈ {x_sel:.4f}")
    figCt = go.Figure()
    if st.session_state.get("has_model", False) and U_num.shape == (len(Xv), len(Tv)):
        figCt.add_trace(go.Scatter(x=Tv, y=U2d[x_idx_fdm, :], mode="lines", name="PINN"))
    figCt.add_trace(go.Scatter(x=Tv, y=U_num[x_idx_fdm, :], mode="lines", name="FDM", line=dict(dash="dash")))
    figCt.update_layout(xaxis_title="t", yaxis_title="T(t)", height=400, title=f"Срез по t при x≈{x_sel:.4f}")
    st.plotly_chart(figCt, use_container_width=True)
    if st.session_state.get("has_model", False) and U_num.shape == (len(Xv), len(Tv)):
        err_slice_t = _err_scalar(U2d[x_idx_fdm, :], U_num[x_idx_fdm, :], err_label)
        st.metric(f"Ошибка на срезе T(t) при x≈{x_sel:.4f} ({err_label})", f"{err_slice_t:.3e}")
# Экспорт CSV всей поверхности
    rows = []
    for i, x in enumerate(Xv):
        for j, t in enumerate(Tv):
            rows.append((x, t, U2d[i, j]))
    df = pd.DataFrame(rows, columns=["x", "t", "T"])
    st.download_button("💾 Скачать CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="pinn_T_result.csv", mime="text/csv")

render_help()











