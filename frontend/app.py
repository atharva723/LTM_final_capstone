# ─────────────────────────────────────────────────────
# frontend/app.py  —  Manufacturing Assistant UI
#
# RUN:  streamlit run frontend/app.py
#
# TABS:
#   1. 💬 Chat Assistant
#   2. 📊 Sensor Dashboard
#   3. 🔧 Fault Diagnostics
#   4. 🛡 Safety & LOTO
#   5. 🔩 Spare Parts
#   6. 🗓 Maintenance Planner
#   7. 📈 Production KPIs
#   8. 📁 Log Analyzer
# ─────────────────────────────────────────────────────

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Page config — must be first Streamlit call ────────
st.set_page_config(
    page_title = "MAIA — Manufacturing AI Assistant",
    page_icon  = "🏭",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Tool imports ──────────────────────────────────────
from backend.tools.sensor_fetch   import get_sensor_data, get_sensor_summary, get_all_sensors
from backend.tools.fault_diagnose import diagnose_fault
from backend.tools.spare_parts    import lookup_parts_by_machine, get_low_stock_parts, search_parts
from backend.tools.maintenance    import calculate_pm_due, format_pm_report
from backend.tools.log_analyzer   import analyze_log, load_uploaded_log, detect_anomalies
from backend.tools.safety_checker import check_safety
from backend.tools.metrics        import compute_oee, get_fleet_metrics
from backend.tools.escalation     import evaluate_and_escalate, get_active_alerts, get_alert_summary, acknowledge_alert
from backend.agent                import get_agent
from config.settings              import MACHINES


# ══════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════
def init_state():
    defaults = {
        "chat_history":   [],
        "session_id":     f"session_{int(time.time())}",
        "operator_name":  "Operator",
        "active_machine": "CNC-M01",
        "last_refresh":   datetime.now(),
        "dark_mode":      True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════
# SIDEBAR  — rendered first so dark_mode state is set
#            before the theme dict is built below
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 MAIA")
    st.markdown("**Plant AI Assistant v1.0**")
    st.markdown("---")

    # ── Theme radio (always visible, no CSS hiding) ───
    theme_choice = st.radio(
        "🎨 Theme",
        options=["🌙 Dark", "☀️ Light"],
        index=0 if st.session_state.dark_mode else 1,
        horizontal=True,
    )
    new_dark = (theme_choice == "🌙 Dark")
    if new_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark
        st.rerun()

    st.markdown("---")

    # ── Operator name ─────────────────────────────────
    st.session_state.operator_name = st.text_input(
        "👤 Operator Name", value=st.session_state.operator_name)

    # ── Collapsible machine selector ──────────────────
    with st.expander("🔧 Active Machine — tap to change", expanded=False):
        chosen = st.selectbox(
            "Machine",
            options=list(MACHINES.keys()),
            index=list(MACHINES.keys()).index(st.session_state.active_machine),
            format_func=lambda x: f"{x} — {MACHINES[x][:22]}",
            key="sidebar_machine_select",
            label_visibility="collapsed",
        )
        st.session_state.active_machine = chosen

    st.caption(f"Active: **{st.session_state.active_machine}**")
    st.markdown("---")

    # ── Fleet Status ──────────────────────────────────
    st.markdown("**Fleet Status**")
    try:
        summary = get_sensor_summary()
        for m in summary:
            mid    = m["machine_id"]
            status = m["overall_status"]
            icon   = {"NORMAL": "🟢", "WARNING": "🟡", "ALARM": "🔴"}.get(status, "⚪")
            alerts = f"  ⚠ {m['alert_count']}" if m["alert_count"] > 0 else ""
            st.markdown(f"{icon} **{mid}**{alerts}")
    except Exception:
        st.caption("Sensor data unavailable")

    st.markdown("---")

    # ── Active Alerts ─────────────────────────────────
    st.markdown("**Active Alerts**")
    try:
        alert_sum = get_alert_summary()
        c1, c2 = st.columns(2)
        c1.metric("🔴 Critical", alert_sum["critical"])
        c2.metric("🟡 High",     alert_sum["high"])
        if alert_sum["unacked"] > 0:
            st.warning(f"{alert_sum['unacked']} unacknowledged alerts")
    except Exception:
        st.caption("Alert data unavailable")

    st.markdown("---")
    st.caption(f"Session: `{st.session_state.session_id[:16]}...`")
    st.caption(f"Refreshed: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh Data"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()


# ══════════════════════════════════════════════════════
# THEME  (built after sidebar so dark_mode is settled)
# ══════════════════════════════════════════════════════
if st.session_state.dark_mode:
    theme = {
        "bg":         "#0d1117",
        "sidebar_bg": "#161b22",
        "card_bg":    "#161b22",
        "card_bg2":   "#1c2128",
        "border":     "#21262d",
        "text":       "#e6edf3",
        "text_muted": "#8b949e",
        "input_bg":   "#161b22",
        "btn_bg":     "#21262d",
        "btn_hover":  "#388bfd22",
    }
else:
    theme = {
        "bg":         "#ffffff",
        "sidebar_bg": "#f6f8fa",
        "card_bg":    "#ffffff",
        "card_bg2":   "#f0f4f8",
        "border":     "#d0d7de",
        "text":       "#1f2328",
        "text_muted": "#57606a",
        "input_bg":   "#ffffff",
        "btn_bg":     "#f6f8fa",
        "btn_hover":  "#ddf4ff",
    }


# ══════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════
st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{
    background: {theme['bg']};
    color: {theme['text']};
  }}
  [data-testid="stSidebar"] {{
    background: {theme['sidebar_bg']};
    border-right: 1px solid {theme['border']};
  }}
  [data-testid="stSidebar"] * {{
    color: {theme['text']};
  }}
  .kpi-card {{
    background: {theme['card_bg']};
    border: 1px solid {theme['border']};
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color .2s;
  }}
  .kpi-card:hover {{ border-color: #388bfd; }}
  .kpi-value {{ font-size: 2rem; font-weight: 700; font-family: 'Courier New', monospace; }}
  .kpi-label {{ font-size: .75rem; color: {theme['text_muted']}; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}
  .kpi-green  {{ color: #3fb950; }}
  .kpi-yellow {{ color: #d29922; }}
  .kpi-red    {{ color: #f85149; }}
  .kpi-blue   {{ color: #388bfd; }}

  .badge        {{ display:inline-block; padding:2px 10px; border-radius:20px; font-size:.75rem; font-weight:600; }}
  .badge-normal {{ background:#1a4a1a; color:#3fb950; border:1px solid #3fb950; }}
  .badge-warn   {{ background:#3d2e00; color:#d29922; border:1px solid #d29922; }}
  .badge-alarm  {{ background:#3d1a1a; color:#f85149; border:1px solid #f85149; }}
  .badge-ok     {{ background:#1a2a4a; color:#388bfd; border:1px solid #388bfd; }}

  .msg-user      {{ background:{theme['card_bg2']}; border-left:3px solid #388bfd; padding:12px 16px; border-radius:8px; margin:8px 0; }}
  .msg-assistant {{ background:{theme['card_bg']}; border-left:3px solid #3fb950; padding:12px 16px; border-radius:8px; margin:8px 0; }}
  .msg-label     {{ font-size:.7rem; color:{theme['text_muted']}; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }}

  .alert-critical {{ background:#3d1a1a; border:1px solid #f85149; border-radius:8px; padding:10px 16px; margin:6px 0; }}
  .alert-high     {{ background:#3d2a1a; border:1px solid #d29922; border-radius:8px; padding:10px 16px; margin:6px 0; }}
  .alert-medium   {{ background:#1a2a3d; border:1px solid #388bfd; border-radius:8px; padding:10px 16px; margin:6px 0; }}

  .section-header {{
    font-size:.7rem; color:{theme['text_muted']}; text-transform:uppercase;
    letter-spacing:2px; border-bottom:1px solid {theme['border']};
    padding-bottom:6px; margin:16px 0 12px 0;
  }}

  .sev-critical {{ color:#f85149; font-weight:700; }}
  .sev-high     {{ color:#d29922; font-weight:700; }}
  .sev-medium   {{ color:#388bfd; font-weight:600; }}
  .sev-low      {{ color:#3fb950; font-weight:600; }}

  .stTextInput > div > div > input {{ background:{theme['input_bg']}; color:{theme['text']}; border-color:{theme['border']}; }}
  .stSelectbox > div > div        {{ background:{theme['input_bg']}; color:{theme['text']}; }}
  div[data-testid="stMetricValue"] {{ color:{theme['text']}; }}
  .stButton > button {{
    background:{theme['btn_bg']}; color:{theme['text']}; border:1px solid {theme['border']};
    border-radius:6px; transition:all .2s;
  }}
  .stButton > button:hover {{ background:{theme['btn_hover']}; border-color:#388bfd; color:#388bfd; }}

  #MainMenu {{ visibility: hidden; }}
 
  footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════
STATUS_COLOR = {"NORMAL": "kpi-green", "WARNING": "kpi-yellow", "ALARM": "kpi-red", "RUNNING": "kpi-green"}
BADGE_CLASS  = {"NORMAL": "badge-normal", "WARNING": "badge-warn", "ALARM": "badge-alarm", "OK": "badge-ok"}

def badge(text, level="NORMAL"):
    cls = BADGE_CLASS.get(level.upper(), "badge-ok")
    return f'<span class="badge {cls}">{text}</span>'

def severity_chip(sev):
    cls = f"sev-{sev.lower()}"
    return f'<span class="{cls}">⬤ {sev}</span>'

def kpi_card(label, value, color="blue", suffix=""):
    color_cls = f"kpi-{color}"
    return f"""
    <div class="kpi-card">
      <div class="kpi-value {color_cls}">{value}{suffix}</div>
      <div class="kpi-label">{label}</div>
    </div>"""

def oee_color(pct):
    if pct >= 85: return "green"
    if pct >= 70: return "yellow"
    return "red"

def sensor_gauge(label, value, min_v, max_v, warn, alarm, unit=""):
    color = "#3fb950" if value < warn else ("#d29922" if value < alarm else "#f85149")
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": f"{label}<br><span style='font-size:.7em;color:#8b949e'>{unit}</span>",
                 "font": {"color": theme['text'], "size": 13}},
        number = {"font": {"color": color, "size": 22}, "suffix": f" {unit}"},
        gauge  = {
            "axis":  {"range": [min_v, max_v], "tickcolor": theme['text_muted'],
                      "tickfont": {"size": 9, "color": theme['text_muted']}},
            "bar":   {"color": color, "thickness": 0.3},
            "bgcolor": theme['card_bg'],
            "bordercolor": theme['border'],
            "steps": [
                {"range": [min_v, warn],  "color": "#1a4a1a"},
                {"range": [warn,  alarm], "color": "#3d2e00"},
                {"range": [alarm, max_v], "color": "#3d1a1a"},
            ],
            "threshold": {"line": {"color": "#f85149", "width": 2}, "value": alarm},
        },
    ))
    fig.update_layout(
        height=180, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=theme['bg'], font_color=theme['text'],
    )
    return fig


# ══════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════
tabs = st.tabs([
    "💬 Chat",
    "📊 Sensor Dashboard",
    "🔧 Fault Diagnostics",
    "🛡 Safety & LOTO",
    "🔩 Spare Parts",
    "🗓 Maintenance",
    "📈 Production KPIs",
    "📁 Log Analyzer",
])


# ══════════════════════════════════════════════════════
# TAB 1 — CHAT ASSISTANT
# ══════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 💬 Chat with MAIA")
    st.caption("Ask anything about machines, faults, safety procedures, maintenance, or production.")

    col_chat, col_ctx = st.columns([2, 1])

    with col_chat:
        chat_container = st.container(height=480)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align:center; color:#8b949e; padding:40px 0;">
                  <div style="font-size:2.5rem">🤖</div>
                  <div style="font-size:1.1rem; margin-top:8px;">Hello! I'm MAIA.</div>
                  <div style="font-size:.85rem; margin-top:4px;">
                    Ask me about sensors, faults, safety, maintenance, or parts.
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="msg-user">
                          <div class="msg-label">👤 {msg.get('operator','Operator')}</div>
                          {msg['content']}
                        </div>""", unsafe_allow_html=True)
                    else:
                        tools_used = msg.get("tools_used", [])
                        tool_tags  = " ".join(
                            f'<span style="background:{theme["card_bg2"]};border:1px solid {theme["border"]};'
                            f'border-radius:4px;padding:1px 6px;font-size:.65rem;'
                            f'color:{theme["text_muted"]};">{t}</span>'
                            for t in tools_used
                        )
                        st.markdown(f"""
                        <div class="msg-assistant">
                          <div class="msg-label">🤖 MAIA {tool_tags}</div>
                          {msg['content'].replace(chr(10), '<br>')}
                        </div>""", unsafe_allow_html=True)

        # st.markdown('<div class="section-header">Quick Actions</div>', unsafe_allow_html=True)
        # qcols = st.columns(4)
        # quick_prompts = [
        #     ("📡 Sensor Status",   f"Show me current sensor status of {st.session_state.active_machine}"),
        #     ("🔧 Fault Help",      f"Error E01 on {st.session_state.active_machine} — what do I do?"),
        #     ("🛡 Safety Briefing", f"What PPE do I need for maintenance on {st.session_state.active_machine}?"),
        #     ("🗓 PM Status",       f"Is maintenance overdue on {st.session_state.active_machine}?"),
        # ]
        # for col, (label, prompt) in zip(qcols, quick_prompts):
        #     if col.button(label, use_container_width=True):
        #         st.session_state["_pending_chat"] = prompt

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Message",
                value       = st.session_state.pop("_pending_chat", ""),
                placeholder = "Ask about faults, sensors, maintenance, safety...",
                label_visibility = "collapsed",
            )
            send_col, clear_col = st.columns([5, 1])
            submitted = send_col.form_submit_button("Send ↵", use_container_width=True, type="primary")
            if clear_col.form_submit_button("Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if submitted and user_input.strip():
            st.session_state.chat_history.append({
                "role":     "user",
                "content":  user_input,
                "operator": st.session_state.operator_name,
            })
            with st.spinner("MAIA is thinking..."):
                try:
                    agent  = get_agent(st.session_state.session_id)
                    result = agent.chat(
                        user_message  = user_input,
                        operator_name = st.session_state.operator_name,
                    )
                    response   = result["response"]
                    tools_used = result.get("tools_used", [])
                    new_alerts = result.get("alerts", [])
                except Exception as e:
                    response   = f"⚠ Agent error: {e}"
                    tools_used = []
                    new_alerts = []

            st.session_state.chat_history.append({
                "role":       "assistant",
                "content":    response,
                "tools_used": tools_used,
            })
            if new_alerts:
                for alert in new_alerts:
                    if alert.get("escalated"):
                        st.warning(f"🚨 Alert escalated: {alert.get('message','')[:80]}")
            st.rerun()

    with col_ctx:
        st.markdown('<div class="section-header">Session Context</div>', unsafe_allow_html=True)
        try:
            agent = get_agent(st.session_state.session_id)
            if agent._initialized:
                state = agent.get_session_state()
                em    = state["entity_memory"]
                st.markdown(f"**Machine:** `{em.get('machine_id') or 'Not set'}`")
                st.markdown(f"**Operator:** `{em.get('operator_name') or 'Unknown'}`")
                st.markdown(f"**Turn:** {state['turn_count']}")
                if em.get("fault_codes"):
                    st.markdown(f"**Faults discussed:** `{', '.join(em['fault_codes'])}`")
                if em.get("attempted_steps"):
                    st.markdown("**Steps attempted:**")
                    for s in em["attempted_steps"][-3:]:
                        st.caption(f"  • {s}")
            else:
                st.caption("Session not started yet.")
        except Exception:
            st.caption("Session context unavailable.")

        st.markdown('<div class="section-header">Recent Alerts</div>', unsafe_allow_html=True)
        try:
            alerts = get_active_alerts(unacknowledged_only=True)[:5]
            if not alerts:
                st.caption("No active alerts")
            for a in alerts:
                sev = a["severity"]
                cls = {"CRITICAL": "alert-critical", "HIGH": "alert-high"}.get(sev, "alert-medium")
                st.markdown(f"""
                <div class="{cls}">
                  <strong>{sev}</strong> — {a['machine_id']}<br>
                  <small>{a['message'][:60]}</small>
                </div>""", unsafe_allow_html=True)
        except Exception:
            st.caption("Alerts unavailable")


# ══════════════════════════════════════════════════════
# TAB 2 — SENSOR DASHBOARD
# ══════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 📊 Real-Time Sensor Dashboard")

    sel_machine = st.selectbox(
        "Select Machine",
        list(MACHINES.keys()),
        format_func = lambda x: f"{x} — {MACHINES[x]}",
        key         = "sensor_machine",
        index       = list(MACHINES.keys()).index(st.session_state.active_machine),
    )

    try:
        data = get_sensor_data(sel_machine, use_cache=False)
    except Exception as e:
        st.error(f"Failed to fetch sensor data: {e}")
        data = {}

    if data and "error" not in data:
        status   = data["overall_status"]
        st_color = {"NORMAL": "green", "WARNING": "yellow", "ALARM": "red"}.get(status, "blue")
        st_icon  = {"NORMAL": "✅", "WARNING": "⚠️", "ALARM": "🚨"}.get(status, "ℹ️")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi_card("Overall Status", f"{st_icon} {status}", st_color), unsafe_allow_html=True)
        k2.markdown(kpi_card("Operator",       data.get("operator","?")[:15],  "blue"), unsafe_allow_html=True)
        k3.markdown(kpi_card("Shift",          data.get("shift","?")[:18],     "blue"), unsafe_allow_html=True)
        k4.markdown(kpi_card("Total Hours",    f"{data.get('operating_hours',0):,}", "blue"), unsafe_allow_html=True)
        k5.markdown(kpi_card("Since PM",       f"{data.get('hours_since_last_pm',0)}h",
                             "yellow" if data.get("hours_since_last_pm",0) > 400 else "green"), unsafe_allow_html=True)

        st.markdown("")

        error_codes = data.get("active_error_codes", [])
        if error_codes:
            st.error(f"🚨 Active Error Codes: **{', '.join(error_codes)}**")

        st.markdown('<div class="section-header">Sensor Readings</div>', unsafe_allow_html=True)

        GAUGE_CONFIGS = {
            "CNC-M01": [
                ("Temperature",  "temperature_c",     20, 100, 65,  80,  "°C"),
                ("Vibration",    "vibration_mm_s",      0,   8, 3.5, 5.0, "mm/s"),
                ("Spindle RPM",  "spindle_rpm",          0,9500,8200,8500, "RPM"),
                ("Coolant Flow", "coolant_flow_l_min",  0,  20,  6,   3,  "L/min"),
                ("Power",        "power_kw",             0,  25, 18,  22,  "kW"),
            ],
            "HYD-P02": [
                ("Pressure",      "pressure_bar",        0, 320,260,280, "bar"),
                ("Temperature",   "temperature_c",      20, 100, 65,  75, "°C"),
                ("Flow Rate",     "flow_l_min",          0, 100, 15,   8, "L/min"),
                ("Motor Current", "motor_current_a",     0,  70, 45,  55, "A"),
                ("Vibration",     "vibration_mm_s",       0,  10,4.0, 6.0,"mm/s"),
            ],
            "CVB-003": [
                ("Belt Speed",    "belt_speed_m_s",      0,   3,1.6, 1.8, "m/s"),
                ("Motor Temp",    "motor_temp_c",       20, 120, 75,  85, "°C"),
                ("Motor Current", "motor_current_a",     0,  45, 28,  35, "A"),
                ("Belt Tension",  "belt_tension_n",      0,1600,600, 400, "N"),
                ("Vibration",     "vibration_mm_s",       0,   8,3.0, 5.0,"mm/s"),
            ],
        }
        configs  = GAUGE_CONFIGS.get(sel_machine, [])
        readings = data.get("sensor_readings", {})

        if configs:
            gcols = st.columns(len(configs))
            for col, (label, key, min_v, max_v, warn, alarm, unit) in zip(gcols, configs):
                if key in readings:
                    val = readings[key]["value"]
                    col.plotly_chart(
                        sensor_gauge(label, val, min_v, max_v, warn, alarm, unit),
                        use_container_width=True,
                    )
                    status_val = readings[key]["status"]
                    col.markdown(
                        f'<center>{badge(status_val, status_val)}</center>',
                        unsafe_allow_html=True,
                    )
        else:
            rows = []
            for sensor, info in readings.items():
                rows.append({"Sensor": sensor, "Value": info["value"], "Status": info["status"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        alerts = data.get("alerts", [])
        if alerts:
            st.markdown('<div class="section-header">Active Sensor Alerts</div>', unsafe_allow_html=True)
            for a in alerts:
                cls = "alert-critical" if a["severity"] == "ALARM" else "alert-high"
                st.markdown(f"""
                <div class="{cls}">
                  <strong>{a['severity']}</strong> — {a['sensor']}: {a['message']}
                </div>""", unsafe_allow_html=True)

        st.caption(f"Last updated: {data.get('timestamp','unknown')}  |  "
                   f"From cache: {data.get('from_cache', False)}")


# ══════════════════════════════════════════════════════
# TAB 3 — FAULT DIAGNOSTICS
# ══════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 🔧 Fault Diagnostics")

    fd_col1, fd_col2 = st.columns([1, 1])
    with fd_col1:
        fd_machine = st.selectbox("Machine", list(MACHINES.keys()),
            format_func=lambda x: f"{x} — {MACHINES[x]}", key="fd_machine")
        fd_code    = st.text_input("Error Code (optional)", placeholder="e.g. E01")
    with fd_col2:
        st.markdown("**Override Sensor Readings (optional)**")
        override_temp = st.number_input("Temperature (°C)", value=0.0, step=0.5)
        override_vib  = st.number_input("Vibration (mm/s)", value=0.0, step=0.1)

    if st.button("🔍 Run Diagnosis", type="primary"):
        sensor_override = {}
        if override_temp > 0: sensor_override["temperature_c"]  = override_temp
        if override_vib  > 0: sensor_override["vibration_mm_s"] = override_vib

        result = diagnose_fault(
            machine_id      = fd_machine,
            error_code      = fd_code.upper() if fd_code else None,
            sensor_readings = sensor_override or None,
        )

        st.markdown('<div class="section-header">Diagnosis Result</div>', unsafe_allow_html=True)
        sev        = result.get("severity", "LOW")
        sev_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🔵", "LOW": "🟢"}
        st.markdown(f"**Severity:** {sev_colors.get(sev,'')} {sev}  —  "
                    f"*{result.get('severity_guidance','')}*")

        if result.get("escalate"):
            st.error("⚠️ **ESCALATION REQUIRED** — Notify supervisor immediately!")

        for diag in result.get("diagnoses", []):
            with st.expander(f"🔎 {diag.get('title','Fault')} — {diag.get('source','').upper()}"):
                st.markdown(f"**Failure Mode:** {diag.get('failure_mode','')}")
                st.markdown("**First-Level Steps:**")
                for i, step in enumerate(diag.get("steps", []), 1):
                    st.markdown(f"{i}. {step}")

        if result.get("recommended_parts"):
            st.markdown('<div class="section-header">Recommended Spare Parts</div>', unsafe_allow_html=True)
            from backend.tools.spare_parts import lookup_parts_by_ids
            parts = lookup_parts_by_ids(result["recommended_parts"])
            if parts:
                df = pd.DataFrame([{
                    "Part ID":     p["part_id"],
                    "Name":        p["name"],
                    "Stock":       p["stock_qty"],
                    "Price (₹)":   p["unit_price"],
                    "Supplier":    p["supplier"],
                    "Lead (days)": p["lead_time_days"],
                } for p in parts])
                st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Error Code Quick Reference</div>', unsafe_allow_html=True)
    from backend.tools.fault_diagnose import ERROR_CODE_DB
    machine_codes = {k: v for k, v in ERROR_CODE_DB.items()
                     if v["machine"] == st.session_state.active_machine}
    if machine_codes:
        ref_rows = [{"Code": k, "Title": v["title"], "Severity": v["severity"],
                     "Failure Mode": v["failure_mode"][:60]}
                    for k, v in machine_codes.items()]
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
# TAB 4 — SAFETY & LOTO
# ══════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🛡 Safety & LOTO Procedures")

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        s_machine = st.selectbox("Machine", list(MACHINES.keys()),
            format_func=lambda x: f"{x} — {MACHINES[x]}", key="s_machine")
    with s_col2:
        s_task = st.selectbox("Task Type", [
            "maintenance", "tool change", "hose replacement",
            "electrical", "cleaning", "blowdown", "welding",
            "belt tracking", "roller change", "chemical dosing",
        ])

    if st.button("🛡 Get Safety Briefing", type="primary"):
        safety    = check_safety(s_machine, s_task)
        haz_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(safety["hazard_level"], "⚪")
        st.markdown(f"### {haz_color} Hazard Level: **{safety['hazard_level']}**")

        if safety["permit_required"]:
            st.warning("📋 **Permit to Work (PTW) required** — get supervisor approval before starting.")
        if safety["loto_required"]:
            st.error("🔒 **LOTO MANDATORY** — Lockout/Tagout all energy sources before work.")

        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("#### ✅ Required PPE")
            for ppe in safety["required_ppe"]:
                st.markdown(f"- {ppe}")
            if safety["prohibited_ppe"]:
                st.markdown("#### ❌ Prohibited")
                for p in safety["prohibited_ppe"]:
                    st.markdown(f"- {p}")
        with pc2:
            st.markdown("#### ⚠️ Hazards")
            for h in safety["hazards"]:
                st.markdown(f"- {h}")

        st.markdown("#### 🚨 Critical Warnings")
        for w in safety["special_warnings"]:
            st.warning(w)

        if safety["loto_steps"]:
            st.markdown("#### 🔒 LOTO Steps")
            for step in safety["loto_steps"]:
                st.markdown(f"{step}")


# ══════════════════════════════════════════════════════
# TAB 5 — SPARE PARTS
# ══════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🔩 Spare Parts Catalog")

    sp_col1, sp_col2, sp_col3 = st.columns([1, 1, 1])
    with sp_col1:
        sp_machine = st.selectbox("Filter by Machine", ["ALL"] + list(MACHINES.keys()), key="sp_machine")
    with sp_col2:
        sp_search = st.text_input("Search Parts", placeholder="bearing, filter, seal...")
    with sp_col3:
        show_low_stock = st.checkbox("⚠ Show Low Stock Only", value=False)

    if show_low_stock:
        parts = get_low_stock_parts(sp_machine if sp_machine != "ALL" else None)
    elif sp_search:
        parts = search_parts(sp_search, sp_machine if sp_machine != "ALL" else None)
    else:
        parts = lookup_parts_by_machine(sp_machine) if sp_machine != "ALL" else []
        if not parts and sp_machine == "ALL":
            from backend.tools.spare_parts import _load_catalog
            parts = _load_catalog()

    if parts:
        df_parts = pd.DataFrame([{
            "Part ID":     p["part_id"],
            "Name":        p["name"],
            "Machine":     p["machine_id"],
            "Category":    p["category"],
            "Stock":       p["stock_qty"],
            "Price (₹)":   p["unit_price"],
            "Reorder At":  p["reorder_level"],
            "Supplier":    p["supplier"],
            "Lead (days)": p["lead_time_days"],
        } for p in parts])

        def stock_style(row):
            if row["Stock"] == 0:
                return ["background-color:#3d1a1a"] * len(row)
            elif row["Stock"] <= row["Reorder At"]:
                return ["background-color:#3d2e00"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_parts.style.apply(stock_style, axis=1),
            use_container_width=True, hide_index=True,
        )
        st.caption(f"Showing {len(parts)} parts  |  🔴 = Out of stock  |  🟡 = Low stock")

        total_val = sum(p["stock_qty"] * p["unit_price"] for p in parts)
        out_stock = sum(1 for p in parts if p["stock_qty"] == 0)
        low_stock = sum(1 for p in parts if 0 < p["stock_qty"] <= p["reorder_level"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Stock Value", f"₹{total_val:,.0f}")
        m2.metric("Out of Stock",      out_stock)
        m3.metric("Low Stock",         low_stock)
    else:
        st.info("No parts found. Try changing the filter or search term.")


# ══════════════════════════════════════════════════════
# TAB 6 — MAINTENANCE PLANNER
# ══════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 🗓 Maintenance Planner")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        m_machine = st.selectbox("Machine", list(MACHINES.keys()),
            format_func=lambda x: f"{x} — {MACHINES[x]}", key="m_machine")
    with mc2:
        m_hours   = st.number_input("Current Operating Hours", min_value=0, value=5000, step=100)
    with mc3:
        m_lphours = st.number_input("Hours at Last PM (0 = auto)", min_value=0, value=0, step=100)

    if st.button("📅 Calculate PM Schedule", type="primary"):
        pm      = calculate_pm_due(
            machine_id    = m_machine,
            current_hours = m_hours,
            last_pm_hours = m_lphours if m_lphours > 0 else None,
        )
        urgency = pm["urgency"]
        if "OVERDUE" in urgency:
            st.error(f"⚠️ {urgency}")
        elif "DUE" in urgency:
            st.warning(f"🔔 {urgency}")
        else:
            st.success(f"✅ {urgency}")

        st.markdown('<div class="section-header">PM Schedule Overview</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(pm["schedule"]), use_container_width=True, hide_index=True)

        if pm["levels_due"]:
            st.markdown('<div class="section-header">Tasks Due Now</div>', unsafe_allow_html=True)
            for lvl in pm["levels_due"]:
                tag = "⚠️ OVERDUE" if lvl["is_overdue"] else "✅ DUE"
                with st.expander(f"{tag} — {lvl['label']} (every {lvl['interval_hours']}h)"):
                    for task in lvl["checklist"]:
                        st.checkbox(task, key=f"task_{lvl['level']}_{task[:20]}")


# ══════════════════════════════════════════════════════
# TAB 7 — PRODUCTION KPIs
# ══════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("## 📈 Production KPIs — OEE Dashboard")

    try:
        fleet = get_fleet_metrics()
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
        fleet = []

    if fleet:
        machine_ids = [m["machine_id"] for m in fleet]
        oee_pcts    = [m["oee"]["score_pct"] for m in fleet]
        avail_pcts  = [m["availability"]["score_pct"] for m in fleet]
        perf_pcts   = [m["performance"]["score_pct"] for m in fleet]
        qual_pcts   = [m["quality"]["score_pct"] for m in fleet]

        fig_fleet = go.Figure()
        fig_fleet.add_trace(go.Bar(name="OEE",          x=machine_ids, y=oee_pcts,   marker_color="#388bfd", opacity=0.9))
        fig_fleet.add_trace(go.Bar(name="Availability", x=machine_ids, y=avail_pcts, marker_color="#3fb950", opacity=0.7))
        fig_fleet.add_trace(go.Bar(name="Performance",  x=machine_ids, y=perf_pcts,  marker_color="#d29922", opacity=0.7))
        fig_fleet.add_trace(go.Bar(name="Quality",      x=machine_ids, y=qual_pcts,  marker_color="#a371f7", opacity=0.7))
        fig_fleet.add_hline(y=85, line_dash="dash", line_color="#f85149",
                             annotation_text="World Class (85%)")
        fig_fleet.update_layout(
            title        = "Fleet OEE Breakdown",
            barmode      = "group",
            height       = 380,
            paper_bgcolor= theme['bg'], plot_bgcolor=theme['bg'],
            font_color   = theme['text'],
            legend       = dict(bgcolor=theme['card_bg'], bordercolor=theme['border']),
            xaxis        = dict(gridcolor=theme['border']),
            yaxis        = dict(gridcolor=theme['border'], range=[0, 105]),
        )
        st.plotly_chart(fig_fleet, use_container_width=True)

        st.markdown('<div class="section-header">Machine KPIs</div>', unsafe_allow_html=True)
        for m in fleet:
            oee_pct = m["oee"]["score_pct"]
            color   = oee_color(oee_pct)
            with st.expander(
                f"**{m['machine_id']}** — {m['machine_name']}  |  "
                f"OEE: {oee_pct:.1f}%  |  Downtime: {m['availability']['downtime_pct']:.1f}%"
            ):
                cols = st.columns(5)
                cols[0].markdown(kpi_card("OEE",          f"{oee_pct:.1f}",                              color,  "%"), unsafe_allow_html=True)
                cols[1].markdown(kpi_card("Availability",  f"{m['availability']['score_pct']:.1f}",      "blue", "%"), unsafe_allow_html=True)
                cols[2].markdown(kpi_card("Performance",   f"{m['performance']['score_pct']:.1f}",       "blue", "%"), unsafe_allow_html=True)
                cols[3].markdown(kpi_card("Quality",       f"{m['quality']['score_pct']:.1f}",           "blue", "%"), unsafe_allow_html=True)
                cols[4].markdown(kpi_card("Downtime",      f"{m['availability']['downtime_pct']:.1f}",
                                         "red" if m['availability']['downtime_pct'] > 15 else "yellow", "%"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 8 — LOG ANALYZER
# ══════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("## 📁 Machine Log Analyzer")

    la_col1, la_col2 = st.columns([1, 2])
    with la_col1:
        la_machine    = st.selectbox("Machine", list(MACHINES.keys()),
            format_func=lambda x: f"{x} — {MACHINES[x]}", key="la_machine")
        uploaded_file = st.file_uploader("Upload CSV Log (optional)", type=["csv"])
        use_stored    = st.checkbox("Use stored 7-day log", value=True)

        if st.button("📊 Analyze Log", type="primary"):
            with st.spinner("Analyzing log data..."):
                file_content = None
                if uploaded_file:
                    file_content = uploaded_file.read()
                    use_stored   = False
                result = analyze_log(la_machine, file_content if not use_stored else None)
                st.session_state["_la_result"] = result

    with la_col2:
        result = st.session_state.get("_la_result")
        if result and "error" not in result:
            oee  = result["oee_stats"]
            anom = result["anomaly_summary"]
            flt  = result["fault_patterns"]

            score    = result["health_score"]
            hs_color = "#3fb950" if score >= 70 else ("#d29922" if score >= 50 else "#f85149")
            fig_health = go.Figure(go.Indicator(
                mode   = "gauge+number",
                value  = score,
                title  = {"text": "Machine Health Score", "font": {"color": theme['text'], "size": 14}},
                number = {"font": {"color": hs_color, "size": 36}, "suffix": "/100"},
                gauge  = {
                    "axis":  {"range": [0, 100], "tickcolor": theme['text_muted']},
                    "bar":   {"color": hs_color},
                    "bgcolor": theme['card_bg'], "bordercolor": theme['border'],
                    "steps": [
                        {"range": [0,  50], "color": "#3d1a1a"},
                        {"range": [50, 70], "color": "#3d2e00"},
                        {"range": [70,100], "color": "#1a4a1a"},
                    ],
                },
            ))
            fig_health.update_layout(
                height=220, margin=dict(l=20, r=20, t=50, b=10),
                paper_bgcolor=theme['bg'],
            )
            st.plotly_chart(fig_health, use_container_width=True)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Availability", f"{oee.get('availability_pct',0):.1f}%")
            k2.metric("Avg OEE",      f"{oee.get('avg_oee_pct',0):.1f}%")
            k3.metric("Alarm Events", anom.get("alarms", 0))
            k4.metric("Total Faults", flt.get("total_faults", 0))

            top_faults = flt.get("top_errors", [])
            if top_faults:
                st.markdown('<div class="section-header">Top Fault Codes</div>', unsafe_allow_html=True)
                fig_faults = px.bar(
                    pd.DataFrame(top_faults),
                    x="code", y="count",
                    color_discrete_sequence=["#f85149"],
                    title="Error Code Frequency (7 days)",
                )
                fig_faults.update_layout(
                    height=250, paper_bgcolor=theme['bg'],
                    plot_bgcolor=theme['bg'], font_color=theme['text'],
                    xaxis=dict(gridcolor=theme['border']),
                    yaxis=dict(gridcolor=theme['border']),
                )
                st.plotly_chart(fig_faults, use_container_width=True)

            if anom.get("events"):
                st.markdown('<div class="section-header">Anomaly Events (Latest 20)</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(anom["events"]), use_container_width=True, hide_index=True)

        elif result and "error" in result:
            st.error(f"Analysis error: {result['error']}")
        else:
            st.info("Select a machine and click **Analyze Log** to begin.")