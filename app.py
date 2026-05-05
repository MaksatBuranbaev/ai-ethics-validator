"""
app.py — Streamlit UI для LLM Ethics Evaluator.

Запуск:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from clients.api_client import get_api_client

st.set_page_config(page_title="LLM Ethics Evaluator", page_icon="⚖️", layout="wide")

client = get_api_client()


# ---------------------------------------------------------------------------
# Вспомогательные функции визуализации
# ---------------------------------------------------------------------------

def _ihum_color(ihum: float, veto: bool) -> str:
    if veto:
        return "#f44336"
    if ihum >= 0.75:
        return "#4caf50"
    if ihum >= 0.5:
        return "#ff9800"
    return "#f44336"


def _radar_chart(ptox: float, pemp: float, ssem: float, ihum: float) -> go.Figure:
    categories = ["Safety (1−Ptox)", "Empathy (Pemp)", "Semantic validity (Ssem)"]
    values = [round(1 - ptox, 4), round(pemp, 4), round(ssem, 4)]
    cats_closed = categories + [categories[0]]
    vals_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill="toself",
        fillcolor="rgba(46, 117, 182, 0.25)",
        line=dict(color="rgba(46, 117, 182, 0.9)", width=2),
        name="Metrics",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[ihum] * 4, theta=cats_closed, mode="lines",
        line=dict(color="rgba(255, 100, 50, 0.7)", width=1.5, dash="dot"),
        name=f"Ihum = {ihum:.3f}",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=11)),
            angularaxis=dict(tickfont=dict(size=13)),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=40, b=60, l=60, r=60),
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Блоки отображения результатов
# ---------------------------------------------------------------------------

def _display_verdict(res: dict) -> None:
    color = _ihum_color(res["ihum"], res["veto"])
    st.markdown(
        f"<div style='background:{color}22; border-left:5px solid {color};"
        f"padding:12px 18px; border-radius:6px; font-size:1.1rem;'>"
        f"<b>{res['verdict']}</b></div>",
        unsafe_allow_html=True,
    )


def _display_metrics(res: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Safety (1−Ptox)", f"{1 - res['ptox']:.3f}")
    col2.metric("Empathy (Pemp)", f"{res['pemp']:.3f}")
    col3.metric("Semantic validity (Ssem)", f"{res['ssem']:.3f}")
    with col4:
        color = _ihum_color(res["ihum"], res["veto"])
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:0.85rem; color:#888'>Ihum</div>"
            f"<div style='font-size:2rem; font-weight:bold; color:{color}'>"
            f"{res['ihum']:.3f}</div></div>",
            unsafe_allow_html=True,
        )


def _display_toxicity_warning(res: dict) -> None:
    if res["veto"]:
        st.error(f"Veto: Ptox = {res['ptox']:.3f} ≥ tau = {res['tau_tox']}")
    elif res["ptox"] > res["tau_tox"] * 0.7:
        st.warning(f"Ptox = {res['ptox']:.3f} близко к порогу tau = {res['tau_tox']}")


def _display_top_k_phrases(phrases: list) -> None:
    if not phrases:
        return
    st.subheader("Ближайшие фразы корпуса")
    for i, item in enumerate(phrases, 1):
        sim = item["similarity"]
        bar_color = "#4caf50" if sim >= 0.7 else "#ff9800" if sim >= 0.4 else "#9e9e9e"
        st.markdown(
            f"**{i}.** \"{item['phrase']}\"  \n"
            f"<span style='color:{bar_color}'>similarity: {sim:.3f}</span>",
            unsafe_allow_html=True,
        )


def _display_xai(tokens: list[str], importances: list[float]) -> None:
    if not tokens:
        st.info("Не удалось получить объяснение для этого текста.")
        return

    max_abs = max(abs(i) for i in importances) or 1.0
    norm = [i / max_abs for i in importances]

    parts = []
    for tok, imp in zip(tokens, norm):
        if imp > 0:
            intensity = min(255, int(255 * imp))
            color = f"rgb(255, {255 - intensity}, {255 - intensity})"
        else:
            intensity = min(255, int(255 * abs(imp)))
            color = f"rgb({255 - intensity}, 255, {255 - intensity})"
        parts.append(
            f'<span style="background-color:{color}; padding:2px; margin:2px; border-radius:4px;">'
            f'{tok}</span>'
        )

    st.markdown(f"<div style='line-height:2.5;'>{''.join(parts)}</div>", unsafe_allow_html=True)
    st.caption("Зелёный — снижает токсичность, красный — повышает.")


def _display_result(res: dict) -> None:
    _display_verdict(res)
    st.markdown("")
    _display_metrics(res)
    _display_toxicity_warning(res)
    st.divider()
    st.subheader("Радар метрик")
    st.plotly_chart(
        _radar_chart(res["ptox"], res["pemp"], res["ssem"], res["ihum"]),
        use_container_width=True,
    )
    _display_top_k_phrases(res.get("top_k_phrases", []))
    with st.expander("Raw JSON (debug)"):
        st.json(res)


# ---------------------------------------------------------------------------
# Вкладки
# ---------------------------------------------------------------------------

def _tab_single() -> None:
    st.caption("Комплексная оценка одного LLM-ответа")
    st.divider()

    tau = st.slider("Порог токсичности (tau)", 0.05, 0.95, 0.3, 0.05, key="tau_single")
    text = st.text_area("Текст LLM-ответа", height=160, key="text_single")

    col_eval, col_xai = st.columns(2)
    eval_clicked = col_eval.button("Оценить", type="primary", use_container_width=True)
    xai_clicked  = col_xai.button("Объяснить токсичность (XAI)", use_container_width=True)

    if eval_clicked:
        if not text.strip():
            st.warning("Введите текст.")
        else:
            with st.spinner("Оцениваем через API..."):
                try:
                    result = client.evaluate(text.strip(), tau=tau)
                    st.session_state["last_result"] = result
                    st.session_state["last_text"] = text.strip()
                    st.divider()
                    _display_result(result)
                except Exception as exc:
                    st.error(f"Ошибка: {exc}")

    if xai_clicked:
        last_text = st.session_state.get("last_text", "")
        if not last_text:
            st.warning("Сначала нажмите «Оценить».")
        else:
            with st.spinner("Вычисляем важность токенов..."):
                try:
                    xai_data = client.explain(last_text)
                    _display_xai(xai_data["tokens"], xai_data["importances"])
                except Exception as exc:
                    st.error(f"Ошибка XAI: {exc}")


def _tab_pareto() -> None:
    st.header("Анализ нескольких ответов")
    st.markdown("Введите ответы, разделённые пустой строкой, или загрузите .txt файл.")

    texts_input = st.text_area(
        "Ответы (разделяйте пустой строкой):",
        height=200,
        placeholder="Ответ 1\n\nОтвет 2\n\nОтвет 3",
        key="text_pareto",
    )
    uploaded = st.file_uploader("Или загрузите .txt (разделитель — пустая строка)", type=["txt"])
    tau = st.slider("Порог токсичности (tau)", 0.05, 0.95, 0.3, 0.05, key="tau_pareto")

    if not st.button("Построить Парето-фронт", type="primary"):
        return

    # Сбор текстов
    if uploaded:
        raw = uploaded.read().decode("utf-8")
    elif texts_input.strip():
        raw = texts_input
    else:
        st.warning("Введите или загрузите хотя бы один ответ.")
        return

    texts = [block.strip() for block in raw.split("\n\n") if block.strip()]
    if len(texts) < 2:
        st.info("Рекомендуется хотя бы 2 ответа для Парето-фронта.")

    with st.spinner("Оцениваем через API..."):
        try:
            data = client.pareto(texts, tau=tau)
        except Exception as exc:
            st.error(f"Ошибка: {exc}")
            return

    results = data["results"]
    pareto_idx = set(data["pareto_indices"])
    st.success(f"Оценено {len(results)} ответов, Парето-оптимальных: {len(pareto_idx)}")

    df = pd.DataFrame([
        {
            "Index": i,
            "Safety (1−Ptox)": 1 - r["ptox"],
            "Empathy (Pemp)": r["pemp"],
            "Semantic validity (Ssem)": r["ssem"],
            "Ihum": r["ihum"],
            "Pareto": i in pareto_idx,
            "Text (short)": (t[:80] + "...") if len(t) > 80 else t,
        }
        for i, (r, t) in enumerate(zip(results, texts))
    ])

    fig = px.scatter_3d(
        df,
        x="Safety (1−Ptox)", y="Empathy (Pemp)", z="Semantic validity (Ssem)",
        color="Pareto",
        color_discrete_map={True: "red", False: "blue"},
        hover_data=["Ihum", "Text (short)"],
        title="Пространство критериев (Парето-фронт — красный)",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Детальные результаты")
    st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("⚖️ LLM Ethics Evaluator")

    # Проверка связи с сервером
    try:
        client.evaluate("test")
    except ConnectionError:
        st.error("Нет связи с API. Запустите сервер: `python server.py`")
        st.stop()
    except Exception:
        pass  # сервер доступен, но текст мог вернуть ошибку — нормально

    tab1, tab2 = st.tabs(["Одиночная оценка", "Парето-фронт"])
    with tab1:
        _tab_single()
    with tab2:
        _tab_pareto()


if __name__ == "__main__":
    main()
