# src/help_tab.py
import platform
import streamlit as st

def _torch_info():
    try:
        import torch
    except Exception as e:
        st.error(f"PyTorch не найден или не импортируется: {e}")
        st.markdown("Установите CPU-версию: `pip install torch --index-url https://download.pytorch.org/whl/cpu` или подберите команду для CUDA на сайте PyTorch.")
        return
    info = {
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    try:
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        info["mps_available"] = False
    st.success("PyTorch найден.")
    st.json(info, expanded=True)

def render_help():
    with st.expander("Справка / Help", expanded=False):
        tab_guide, tab_diag, tab_cfg = st.tabs(["Быстрый старт", "Диагностика", "Текущая конфигурация"])

        # ===== Быстрый старт
        with tab_guide:
            st.markdown(
                "### Установка (кратко)\n"
                "• Windows: создайте venv в %USERPROFILE%\\\\venvs\\\\pinn-heat, активируйте, затем установите: `pip install streamlit plotly pandas numpy` и PyTorch (CPU или CUDA).\n"
                "• macOS/Linux: venv в ~/venvs/pinn-heat, активируйте, установите пакеты аналогично.\n\n"
                "### Запуск\n"
                "`streamlit run app.py`"
            )

        # ===== Диагностика
        with tab_diag:
            st.markdown("### Проверка PyTorch / устройства")
            if st.button("Проверить PyTorch/устройство", key="btn_help_check_torch"):
                _torch_info()
            st.markdown("---")
            st.markdown("### Быстрые команды восстановления")
            st.code(
                "python -m pip install --upgrade pip wheel\n"
                "pip install --upgrade streamlit plotly pandas numpy\n"
                "pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu\n",
                language="bash"
            )

        # ===== Текущая конфигурация
        with tab_cfg:
            conf = {
                "model_name": st.session_state.get("model_name"),
                "params": {
                    "kappa": st.session_state.get("kappa"),
                    "tau_R": st.session_state.get("tau_R"),
                    "kappa_tilde": st.session_state.get("kappa_tilde"),
                },
                "domain": {
                    "x_min": st.session_state.get("x_min"),
                    "x_max": st.session_state.get("x_max"),
                    "t_min": st.session_state.get("t_min"),
                    "t_max": st.session_state.get("t_max"),
                },
                "grid": {"nx": st.session_state.get("nx"), "nt": st.session_state.get("nt")},
                "boundary_conditions": {
                    "left_type": st.session_state.get("left_bc_type"),
                    "right_type": st.session_state.get("right_bc_type"),
                    "left_expr": st.session_state.get("left_bc_expr"),
                    "right_expr": st.session_state.get("right_bc_expr"),
                },
                "initial_conditions": {
                    "T(x,t_min)": st.session_state.get("ic_expr"),
                    "dT/dt(x,t_min)": st.session_state.get("ict_expr"),
                },
                "source": {"q(x,t)": st.session_state.get("q_expr")},
                "fdm": (st.session_state.get("fdm") or {}),
                "pinn_trained": bool(st.session_state.get("has_model", False)),
            }
            st.json(conf, expanded=True)
            st.caption("Если значения пустые/None — соответствующие поля ещё не заданы или блоки не запускались.")

