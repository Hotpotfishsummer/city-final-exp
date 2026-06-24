"""
================================================================================
深圳市核心商圈 24h 出行潮汐与热点演化大屏 (城市空间计算与大模型决策集成系统)
================================================================================
- 启动入口: `streamlit run app_new.py`(本文件)
- 项目定位: 深圳技术大学《城市多模态数据与大模型应用》期末大作业核心系统
- 系统闭环: 空间多源计算感知 ──► 时空潮汐特征演化 ──► LLM时空语义诊断 (DeepSeek RAG)

[多源空间数据集集成说明]
- data/streets_24h_core.geojson   (28 核心商圈街道 × 24h 流量矩阵 × POI 数据, 空间聚合底座)
- data/street_hour_full.geojson   (74 全市域街道 × 24h 出行 O-D Pickup/Dropoff/Net 三层特征流)
- data/tidal_clusters.csv         (28 核心街道 24h 职住通勤与潮汐特征聚类标签, 阶段③ 产出)
- data/relay_edges.csv            (24h 核心路网时空热点流动接力边数据, 阶段③ 产出)
- data/poi_corr.json              (商圈实体 POI 存量与打车出行峰值的 Pearson/Spearman 相关系数)

[大屏现代化布局与交互交互设计]
- 侧边控制台(左一): 全天 24 小时轴滑动、起点/终点/净流入视角切换、全市/核心区范围筛选、AI决策舱唤醒
- 动态交互地图(中上): 暗色高亮 Choropleth 街道热力底图，支持 hover 动态青色发光边框与 tooltip 悬浮窗
- AI 城市决策舱(右上): 唤醒后拉出 3:7 独立面板，支持专家身份切换(安保/规划)、热点街道点击联动高亮
                          以及基于 DeepSeek API 的 100% 真实 RAG 决策报告流式吐字输出（打字机特效）
- 多维指标面板(下半): 包含潮汐曲线、POI-流量散点耦合分析、24h 核心路网流动接力网络 3 大分页 Tabs 视图
================================================================================
"""
import os
import json
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from openai import OpenAI
from shapely.geometry import Point

# ============================================================
# 0. 大模型配置 (OpenAI 兼容协议,可对接任意兼容 Chat Completions 的服务)
# ============================================================
# 配置方式 (按优先级):
#   1. Streamlit secrets: 在 .streamlit/secrets.toml 写入
#        OPENAI_API_KEY  = "sk-xxx"
#        OPENAI_BASE_URL = ""                   # 留空 = 走 OpenAI 官方
#        OPENAI_MODEL    = "gpt-4o-mini"        # 任意 OpenAI 兼容模型名
#   2. 环境变量: export OPENAI_API_KEY=sk-xxx ...
#   3. 直接修改下方 DEFAULT_* 常量
DEFAULT_API_KEY  = ""
DEFAULT_BASE_URL = ""                  # 留空 = 走 OpenAI 官方
DEFAULT_MODEL    = "gpt-4o-mini"       # OpenAI 兼容格式的模型名

def _get_config(key: str, default: str) -> str:
    """优先从 st.secrets 读,再读环境变量,最后用 default。"""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)

API_KEY  = _get_config("OPENAI_API_KEY",  DEFAULT_API_KEY)
BASE_URL = _get_config("OPENAI_BASE_URL", DEFAULT_BASE_URL)
MODEL    = _get_config("OPENAI_MODEL",    DEFAULT_MODEL)


# ============================================================
# 0. 路径与基础配置
# ============================================================
APP_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(APP_DIR, "data")
HOURS     = list(range(24))
HOURS_ARR = np.asarray(HOURS)            # ⚠️ matplotlib 需要 ndarray,list 会被 np.array 推为 object dtype
HOURS_LBL = [f"{h:02d}:00" for h in HOURS]

# 中文字体 (matplotlib)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="深圳核心商圈 24h 潮汐与热点演化大屏",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ 深圳市核心商圈 24h 出行潮汐与热点演化大屏")
st.caption("数据源:出租车 GPS 订单(2019-09-01)  |  阶段②③ 集成:74 街道 24h 流量矩阵 + 潮汐聚类 + 接力网络")

# ============================================================
# 0.5 AI 助手状态初始化
# ============================================================
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []
if "clicked_street" not in st.session_state:
    st.session_state.clicked_street = None
if "click_counter" not in st.session_state:
    st.session_state.click_counter = 0
if "ai_panel_open" not in st.session_state:
    st.session_state.ai_panel_open = False

# ============================================================
# 1. 数据加载 (用 cache 提速)
# ============================================================
@st.cache_data
def load_core_28():
    """同学聚合的 28 核心街道:带 POI + 24h flow"""
    return gpd.read_file(os.path.join(DATA_DIR, "streets_24h_core.geojson"))

@st.cache_data
def load_full_74():
    """阶段② 产出的 74 街道 × 24h 三层流量"""
    return gpd.read_file(os.path.join(DATA_DIR, "street_hour_full.geojson"))

@st.cache_data
def load_tidal():
    """阶段③ 产出的潮汐聚类标签"""
    return pd.read_csv(os.path.join(DATA_DIR, "tidal_clusters.csv"))

@st.cache_data
def load_relay():
    """阶段③ 产出的接力边"""
    return pd.read_csv(os.path.join(DATA_DIR, "relay_edges.csv"))

@st.cache_data
def load_poi_corr():
    with open(os.path.join(DATA_DIR, "poi_corr.json"), encoding="utf-8") as f:
        return json.load(f)

core = load_core_28()
full = load_full_74()
tidal = load_tidal()
relay = load_relay()
poi_corr = load_poi_corr()

# 兼容旧字段 (streets_24h_core 用 flow_h, full 用 pickup_h)
flow_cols_core = [f"flow_{h}" for h in HOURS]
pick_cols_full = [f"pickup_{h}" for h in HOURS]
drop_cols_full = [f"dropoff_{h}" for h in HOURS]
net_cols_full  = [f"net_{h}"     for h in HOURS]

# 大区融合 (3 核心区) - 用于大区名 / 边界
district_gdf = core.dissolve(by="DISTRICT").reset_index()

# 把潮汐标签 join 回 core (left join)
core = core.merge(tidal[["NAME", "tidal_type"]], on="NAME", how="left")
core["tidal_type"] = core["tidal_type"].fillna("未分类")

# ============================================================
# 2. 侧边栏
# ============================================================
st.sidebar.header("🕹️ 时空控制台")

selected_hour = st.sidebar.slider(
    "⏰ 当前时间切片",
    min_value=0, max_value=23, value=8, step=1,
    format="%d:00",
)

# 视图模式
view_mode = st.sidebar.radio(
    "🌊 流量视角",
    options=["起点流量 (Pickup)", "终点流量 (Dropoff)", "净流入 (Net)"],
    index=0,
    help="起点 = 人们从此处出发;终点 = 人们到达此处;净流入 = 终点 − 起点",
)

# 数据源
data_source = st.sidebar.radio(
    "📂 街道覆盖范围",
    options=["核心 28 街道 (含 POI)", "全市 74 街道"],
    index=0,
    help="28 街道:含 POI 数,可叠加潮汐标签;74 街道:全市域,无 POI 标签",
)

# 大区筛选
districts_all = sorted(core["DISTRICT"].unique().tolist())
selected_dists = st.sidebar.multiselect(
    "🏢 大区筛选", options=districts_all, default=districts_all
)

# 接力开关
show_relay = st.sidebar.checkbox("🎯 显示 top-8 热点接力箭头", value=False)

# 关键指标卡
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 关键指标")
st.sidebar.metric("POI-流量 Pearson r", f"{poi_corr['pearson_poi_vs_max']['r']:.3f}",
                  help="POI 数与流量峰值的相关系数(阶段③ 结论)")
# 优先用 业务特征空间 4 类的 silhouette(更准),为兼容老 json,退而用旧 K=2 值
_sil = poi_corr.get("silhouette_business_K4") or max(poi_corr["silhouette"].values())
st.sidebar.metric("潮汐聚类 silhouette", f"{_sil:.3f}",
                  help="4 类潮汐聚类(业务特征空间)的轮廓系数")
st.sidebar.metric("接力边总数", f"{len(relay)}",
                  help="24h 内 enter/leave/stay 边数合计")

# 补充:异常点指标(显式转 float)
try:
    flow_f = core[flow_cols_core].astype(float)
    _z = (flow_f.max(axis=1) - flow_f.mean(axis=1)).div(flow_f.std(axis=1)).abs()
    n_anom = int((_z > 1.5).sum())
except Exception:
    n_anom = "—"
st.sidebar.metric("异常高流量街道", n_anom,
                  help="24h 峰值 z-score > 1.5 的街道数")

# AI 模型状态 (OpenAI 兼容配置:从 st.secrets / 环境变量读取)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI 模型状态")
if API_KEY:
    _mask = (API_KEY[:4] + "***" + API_KEY[-2:]) if len(API_KEY) > 8 else API_KEY
    st.sidebar.markdown(
        f"- ✅ **已配置**  \n"
        f"- 模型: `{MODEL}`  \n"
        f"- 网关: `{'OpenAI 官方' if not BASE_URL else BASE_URL}`  \n"
        f"- Key: `{_mask}`"
    )
else:
    st.sidebar.warning(
        "未配置 `OPENAI_API_KEY`,AI 决策舱将不可用。\n\n"
        "在 `.streamlit/secrets.toml` 写入或设置同名环境变量后重启。"
    )

# ============================================================
# 3. 数据过滤
# ============================================================
if data_source.startswith("核心"):
    gdf = core[core["DISTRICT"].isin(selected_dists)].copy()
    flow_col = f"flow_{selected_hour}"     # 同学聚合
    view_col = flow_col
    has_tidal = True
    has_poi   = True
    label_set = "core"
else:
    gdf = full[full["DISTRICT"].isin(selected_dists)].copy()
    if "起点" in view_mode:
        view_col = f"pickup_{selected_hour}"
    elif "终点" in view_mode:
        view_col = f"dropoff_{selected_hour}"
    else:
        view_col = f"net_{selected_hour}"
    flow_col = view_col
    has_tidal = False
    has_poi   = False
    label_set = "full"

if len(gdf) == 0:
    st.warning("当前筛选无数据,请检查大区选择。")
    st.stop()

# ============================================================
# 4. 主图:暗色 Folium（每次点击动态重建，保证高亮即时显示）
# ============================================================
_map_deps = (
    selected_hour, view_mode, data_source, tuple(selected_dists),
    show_relay, label_set, st.session_state.clicked_street
)

def build_map(deps):
    """构建完整地图（含高亮图层），每次deps变化时重建"""
    m = folium.Map(
        location=[22.55, 114.05],
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )
    
    # 4.1 Choropleth 热力
    folium.Choropleth(
        geo_data=gdf,
        data=gdf,
        columns=["NAME", view_col],
        key_on="feature.properties.NAME",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.2,
        nan_fill_color="black",
        legend_name=f"{HOURS_LBL[selected_hour]} {view_mode} (车次)",
    ).add_to(m)

    # 4.2 大区边界
    folium.GeoJson(
        district_gdf,
        style_function=lambda x: {
            "color": "#FFD700",
            "weight": 3,
            "fillOpacity": 0,
            "dashArray": "6, 6",
        },
        name="核心区行政边界",
    ).add_to(m)
    
    for _, row in district_gdf.iterrows():
        centroid = row.geometry.centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=f'<div style="color:#FFD700;font-size:22px;font-weight:900;'
                     f'text-shadow:2px 2px 4px #000;letter-spacing:2px;'
                     f'white-space:nowrap;opacity:0.85;">⭐ {row["DISTRICT"]}</div>'
            ),
        ).add_to(m)
    
    # 4.3 当前最热 top-12 街道青色标签
    top_streets = gdf.nlargest(12, view_col)
    for _, row in top_streets.iterrows():
        centroid = row.geometry.centroid
        extra = ""
        if has_tidal and isinstance(row.get("tidal_type"), str):
            extra = f" <span style='color:#FFA500;font-size:10px'>[{row['tidal_type']}]</span>"
        folium.Marker(
            location=[centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=f'<div style="color:#00FFFF;font-size:12px;font-weight:bold;'
                     f'text-shadow:1px 1px 2px #000;white-space:nowrap;">'
                     f'📍{row["NAME"]}{extra}</div>'
            ),
        ).add_to(m)

    # 4.4 接力箭头图层
    if show_relay and label_set == "core":
        cur_hour = selected_hour
        next_hour = (cur_hour + 1) % 24
        cur_top = core.nlargest(8, f"flow_{cur_hour}")["NAME"].tolist()
        next_top = core.nlargest(8, f"flow_{next_hour}")["NAME"].tolist()
        geom = core.set_index("NAME")["geometry"].to_dict()
        for src in cur_top:
            for tgt in next_top:
                if src == tgt:
                    continue
                try:
                    p1 = geom[src].centroid
                    p2 = geom[tgt].centroid
                    folium.PolyLine(
                        locations=[[p1.y, p1.x], [p2.y, p2.x]],
                        color="#FF1493",
                        weight=1.5,
                        opacity=0.5,
                        dash_array="4, 6",
                    ).add_to(m)
                except Exception:
                    pass

    # 4.5 Tooltip 浮窗 + hover 高亮
    if has_poi:
        tip_fields = ["DISTRICT", "NAME", "poi_count", view_col, "tidal_type"]
        tip_aliases = ["🏢 大区:", "📍 街道:", "🏪 POI 数:", f"🔥 {HOURS_LBL[selected_hour]} {view_mode}:", "🌊 潮汐类型:"]
    else:
        tip_fields = ["DISTRICT", "NAME", view_col]
        tip_aliases = ["🏢 大区:", "📍 街道:", f"🔥 {HOURS_LBL[selected_hour]} {view_mode}:"]

    tip = folium.features.GeoJson(
        gdf,
        style_function=lambda x: {"fillColor": "#00000000", "color": "#00000000"},
        highlight_function=lambda x: {
            "color": "#00FFFF",
            "weight": 4,
            "fillColor": "#00FFFF",
            "fillOpacity": 0.08,
            "opacity": 0.9,
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=tip_fields, aliases=tip_aliases,
            style=("background-color:rgba(20,20,20,0.9);color:white;"
                   "border-radius:4px;padding:10px;font-size:13px;"),
        ),
    )
    m.add_child(tip)

    # 4.6 点击街道 → 实际多边形轮廓高亮
    clicked_name = deps[-1]
    if clicked_name:
        clicked_gdf = core[core["NAME"] == clicked_name].copy()
        if len(clicked_gdf) > 0:
            folium.GeoJson(
                clicked_gdf,
                style_function=lambda x: {
                    "color": "#00FFFF",
                    "weight": 5,
                    "fillColor": "#00FFFF",
                    "fillOpacity": 0.08,
                    "opacity": 0.95,
                },
                name="clicked_highlight",
            ).add_to(m)
    
    return m, top_streets

m, top_streets = build_map(_map_deps)

# ============================================================
# 辅助函数：根据地理坐标反查街道名称
# ============================================================
def find_street_by_coords(lat: float, lng: float, gdf: gpd.GeoDataFrame) -> str:
    point = Point(lng, lat)
    best_name = None
    best_dist = float("inf")
    for _, row in gdf.iterrows():
        if row.geometry.contains(point):
            return row["NAME"]
        dist = row.geometry.distance(point)
        if dist < best_dist:
            best_dist = dist
            best_name = row["NAME"]
    if best_dist > 0.1:
        return None
    return best_name


# ============================================================
# AI 面板专用函数：大模型流式调用骨架
# ============================================================
def build_ai_system_prompt(role: str, context: dict) -> str:
    hour_label = HOURS_LBL[context.get("hour", 8)]
    view_mode_label = context.get("view_mode", "起点流量")
    data_source_label = context.get("data_source", "核心 28 街道")
    target_street = context.get("top_street_name", "N/A")
    target_val = context.get("top_street_val", 0)
    avg_val = context.get("all_streets_avg", 0)
    top3_list = context.get("top3_list", [])
    clicked_tidal = context.get("clicked_tidal", "未知")
    clicked_poi = context.get("clicked_poi", "未知")
    has_tidal_flag = context.get("has_tidal", False)
    has_poi_flag = context.get("has_poi", False)

    if avg_val > 0 and target_val > 0:
        ratio = target_val / avg_val
        if ratio <= 0.8:
            level_note = f"【流量评估】{target_street} 当前流量 {target_val} 车次，低于全市均值 {avg_val:.0f} 车次，属于低流量区域。"
        elif ratio <= 1.5:
            level_note = f"【流量评估】{target_street} 当前流量 {target_val} 车次，接近全市均值 {avg_val:.0f} 车次，属于中等流量区域。"
        else:
            level_note = f"【流量评估】{target_street} 当前流量 {target_val} 车次，显著高于全市均值 {avg_val:.0f} 车次，属于高流量热点区域！"
    else:
        level_note = f"【流量评估】{target_street} 当前流量 {target_val} 车次。"

    top3_text = ""
    if top3_list:
        top3_items = "、".join([f"{name}({val} 车次)" for name, val in top3_list])
        top3_text = f"当前全市 Top-3 热点街道：{top3_items}"

    extra_props = ""
    if has_tidal_flag and clicked_tidal != "未知":
        extra_props += f"潮汐类型: {clicked_tidal}；"
    if has_poi_flag and clicked_poi != "未知":
        extra_props += f"POI 数: {clicked_poi}；"
    if extra_props:
        extra_props = f"【{target_street} 属性】{extra_props}"

    base_data = (
        f"【当前实时数据】\n"
        f"- ⏰ 时间切片：{hour_label} | 视图模式：{view_mode_label}\n"
        f"- 📂 数据源：{data_source_label}\n"
        f"- 🎯 关注街道：{target_street}（{target_val} 车次）\n"
        f"{level_note}\n"
        f"{top3_text}\n"
        f"{extra_props}\n"
    )

    if role == "🚨 交通安保大队长":
        return base_data + (
            "\n你是深圳市交通安保大队的 AI 决策专家。请基于以上实时数据进行研判：\n"
            "1. 根据流量量级判断是否需要关注：低流量区域直接说明无需特殊措施；高流量区域才分析踩踏风险、疏散通道压力。\n"
            "2. 如果流量显著偏高，给出街道级管制或限流预案。\n"
            "3. 用清晰的要点表达，必须引用具体数据指标。\n"
            "4. 拒绝模板化输出，每次回答必须因地制宜。"
        )
    else:
        return base_data + (
            "\n你是深圳市城市商业网点规划师，专注商圈盲区识别与新零售选址。请基于以上实时数据进行研判：\n"
            "1. 分析该街道的流量与 POI 配置是否匹配：高流量低 POI 说明有商业机会，低流量高 POI 说明可能饱和。\n"
            "2. 结合潮汐类型判断商业配套是否合理（如午间型区域是否缺少餐饮）。\n"
            "3. 如果流量和 POI 数据都偏低，直接说明该区域当前不具备高潜力，无需强行推荐选址。\n"
            "4. 用清晰的要点表达，必须引用具体数据指标。\n"
            "5. 拒绝模板化输出，每次回答必须因地制宜。"
        )

def stream_ai_response(role: str, user_message: str, context: dict):
    if not API_KEY:
        yield (
            "⚠️ 未配置 OPENAI_API_KEY,无法调用大模型。\n\n"
            "配置方式 (任选其一):\n"
            "  • `.streamlit/secrets.toml`:\n"
            "      OPENAI_API_KEY  = 'sk-xxx'\n"
            "      OPENAI_BASE_URL = ''  # 留空走 OpenAI 官方\n"
            "      OPENAI_MODEL    = 'gpt-4o-mini'\n"
            "  • 环境变量: export OPENAI_API_KEY=sk-xxx ...\n"
            "修改后请重启 Streamlit。\n"
        )
        return
    try:
        # OpenAI 兼容协议: base_url 留空走官方,非空指向第三方网关
        client_kwargs = {"api_key": API_KEY}
        if BASE_URL:
            client_kwargs["base_url"] = BASE_URL
        client = OpenAI(**client_kwargs)
        sys_prompt = build_ai_system_prompt(role, context)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_message},
        ]
        if st.session_state.ai_messages:
            history = st.session_state.ai_messages[-8:]
            for msg in history:
                messages.insert(-1, msg)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1024,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"⚠️ 大模型调用出错:{str(e)}\n\n当前配置: MODEL={MODEL!r}, BASE_URL={BASE_URL or '<OpenAI 官方>'!r}"

# ============================================================
# 4.7 注入全局 CSS（修复全局选择器污染 Bug）
# ============================================================
st.markdown(
    """
    <style>
    @keyframes ai-pulse {
        0% { box-shadow: 0 0 25px rgba(108, 92, 231, 0.6); }
        50% { box-shadow: 0 0 45px rgba(108, 92, 231, 0.9), 0 0 60px rgba(0, 206, 201, 0.4); }
        100% { box-shadow: 0 0 25px rgba(108, 92, 231, 0.6); }
    }
    .ai-hotspot {
        background: rgba(108, 92, 231, 0.15);
        border: 1px solid rgba(108, 92, 231, 0.3);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 14px;
        color: #fff;
        font-size: 14px;
    }
    .ai-hotspot span {
        color: #FFD700;
        font-weight: 600;
    }
    /* 仅针对含有特定特征的关闭按钮进行精确定位美化，消除全局大屏Tab列冲突 */
    div[data-testid="stButton"]:has(button[key="ai_close_panel_btn"]) button {
        background: rgba(255, 80, 80, 0.2) !important;
        border: 1px solid rgba(255, 80, 80, 0.4) !important;
        color: #fff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stButton"]:has(button[key="ai_close_panel_btn"]) button:hover {
        background: rgba(255, 80, 80, 0.4) !important;
    }
    div[data-testid="stColumn"]:nth-child(2) p, 
    div[data-testid="stColumn"]:nth-child(2) label, 
    div[data-testid="stColumn"]:nth-child(2) div {
        color: #1A1A1A !important; 
        font-weight: 600 !important; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 5. 主布局
# ============================================================

def _toggle_ai():
    st.session_state.ai_panel_open = not st.session_state.ai_panel_open

st.markdown(
    """
    <style>
    div[data-testid="stButton"]:has(button[key="ai_float_btn"]) {
        position: fixed !important;
        top: 20px !important;
        right: 20px !important;
        z-index: 9999 !important;
    }
    div[data-testid="stButton"]:has(button[key="ai_float_btn"]) button {
        width: 120px !important;
        height: 48px !important;
        border-radius: 24px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #6C5CE7, #00CEC9) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 0 25px rgba(108,92,231,0.6), 0 4px 15px rgba(0,0,0,0.4) !important;
        animation: ai-pulse 2s infinite !important;
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.columns([1])[0].empty()
_btn_label = "✕ 关闭" if st.session_state.ai_panel_open else "🤖 AI 助手"
st.button(_btn_label, key="ai_float_btn", on_click=_toggle_ai, help="打开/关闭 AI 城市大脑决策面板")

if st.session_state.ai_panel_open:
    # 双栏布局：左侧地图 65%，右侧 AI 面板 35%
    col_map, col_ai = st.columns([6.5, 3.5])
    
    with col_map:
        map_data = st_folium(m, width="100%", height=620, key=f"map_main_{st.session_state.click_counter}")
        if map_data and map_data.get("last_object_clicked"):
            click_lat = map_data["last_object_clicked"]["lat"]
            click_lng = map_data["last_object_clicked"]["lng"]
            clicked_name = find_street_by_coords(click_lat, click_lng, core)
            if clicked_name and clicked_name != st.session_state.clicked_street:
                st.session_state.clicked_street = clicked_name
                st.session_state.click_counter += 1
                st.rerun()

    with col_ai:
        st.markdown("### 🧠 AI 城市大脑决策官")
        st.button("✕ 关闭面板", key="ai_close_panel_btn", on_click=_toggle_ai, width='stretch')
        
        try:
            if st.session_state.clicked_street:
                ai_top_name = st.session_state.clicked_street
                clicked_row = gdf[gdf["NAME"] == ai_top_name]
                ai_top_val = int(clicked_row.iloc[0][view_col]) if len(clicked_row) > 0 else 0
            else:
                ai_top_name = top_streets.iloc[0]["NAME"]
                ai_top_val = int(top_streets.iloc[0][view_col])
        except Exception:
            ai_top_name = "N/A"
            ai_top_val = 0
        
        # 📌 引入固定高度的滚动容器，完美将大模型流式内容与历史记录锁在固定盒子中，不向下撑开全局
        with st.container(height=460):
            st.markdown(
                f'<div class="ai-hotspot">🔥 当前侦测热点：<span>{ai_top_name}</span> | '
                f'瞬时车次：<span>{ai_top_val}</span> 车次</div>',
                unsafe_allow_html=True,
            )
            
            def _on_role_change():
                st.session_state.ai_messages = []
            
            ai_role = st.radio(
                "指派 AI 专家身份",
                options=["🚨 交通安保大队长", "🏗️ 城市网点规划师"],
                index=0,
                key="ai_role_radio_panel",
                on_change=_on_role_change,
            )
            
            if st.button("🚀 现场研判生成报告", type="primary", width='stretch', key="ai_report_btn"):
                all_vals = gdf[view_col].astype(float)
                all_streets_avg = float(all_vals.mean()) if len(all_vals) > 0 else 0
                top3_raw = gdf.nlargest(3, view_col)[["NAME", view_col]]
                top3_list = [(row["NAME"], int(row[view_col])) for _, row in top3_raw.iterrows()]
                clicked_tidal = "未知"
                clicked_poi = "未知"
                if ai_top_name != "N/A":
                    target_row = gdf[gdf["NAME"] == ai_top_name]
                    if len(target_row) > 0:
                        tr = target_row.iloc[0]
                        if has_tidal and "tidal_type" in tr:
                            clicked_tidal = tr["tidal_type"]
                        if has_poi and "poi_count" in tr:
                            clicked_poi = int(tr["poi_count"])
                context = {
                    "hour": selected_hour, "view_mode": view_mode, "data_source": data_source,
                    "top_street_name": ai_top_name, "top_street_val": ai_top_val,
                    "all_streets_avg": all_streets_avg, "top3_list": top3_list,
                    "clicked_tidal": clicked_tidal, "clicked_poi": clicked_poi,
                    "selected_dists": selected_dists, "has_tidal": has_tidal, "has_poi": has_poi,
                }
                with st.spinner("AI 正在空间推理..."):
                    response_text = st.write_stream(
                        stream_ai_response(ai_role, "请基于当前实时数据进行研判分析。", context)
                    )
                st.session_state.ai_messages.append({"role": "user", "content": f"当前时间切片：{HOURS_LBL[selected_hour]}，热点：{ai_top_name}"})
                st.session_state.ai_messages.append({"role": "assistant", "content": response_text})
                st.rerun()
            
            if st.session_state.ai_messages:
                st.markdown("---")
                st.markdown("##### 📜 决策对话历史记录")
                for msg in reversed(st.session_state.ai_messages):
                    role = msg["role"]
                    content_text = msg["content"]
                    if role == "user":
                        st.markdown(f"**🧑‍💼 指挥官:** {content_text}")
                    else:
                        st.markdown(f"**🤖 AI 专家:** {content_text}")
                    st.markdown("<hr style='margin:4px 0;opacity:0.2;'>", unsafe_allow_html=True)
        
        # 📌 将对话追问组件钉在滚动容器外部，始终保持在 AI 面板底端
        followup = st.chat_input("您可以向 AI 专家继续追问应急细节...", key="ai_followup_panel")
        if followup:
            all_vals = gdf[view_col].astype(float)
            all_streets_avg = float(all_vals.mean()) if len(all_vals) > 0 else 0
            top3_raw = gdf.nlargest(3, view_col)[["NAME", view_col]]
            top3_list = [(row["NAME"], int(row[view_col])) for _, row in top3_raw.iterrows()]
            clicked_tidal = "未知"
            clicked_poi = "未知"
            if ai_top_name != "N/A":
                target_row = gdf[gdf["NAME"] == ai_top_name]
                if len(target_row) > 0:
                    tr = target_row.iloc[0]
                    if has_tidal and "tidal_type" in tr:
                        clicked_tidal = tr["tidal_type"]
                    if has_poi and "poi_count" in tr:
                        clicked_poi = int(tr["poi_count"])
            context = {
                "hour": selected_hour, "view_mode": view_mode, "data_source": data_source,
                "top_street_name": ai_top_name, "top_street_val": ai_top_val,
                "all_streets_avg": all_streets_avg, "top3_list": top3_list,
                "clicked_tidal": clicked_tidal, "clicked_poi": clicked_poi,
                "selected_dists": selected_dists, "has_tidal": has_tidal, "has_poi": has_poi,
            }
            with st.spinner("AI 正在深度研判..."):
                response_text = st.write_stream(
                    stream_ai_response(ai_role, followup, context)
                )
            st.session_state.ai_messages.append({"role": "user", "content": followup})
            st.session_state.ai_messages.append({"role": "assistant", "content": response_text})
            st.rerun()
else:
    map_data = st_folium(m, width="100%", height=620, key=f"map_main_{st.session_state.click_counter}")
    if map_data and map_data.get("last_object_clicked"):
        click_lat = map_data["last_object_clicked"]["lat"]
        click_lng = map_data["last_object_clicked"]["lng"]
        clicked_name = find_street_by_coords(click_lat, click_lng, core)
        if clicked_name and clicked_name != st.session_state.clicked_street:
            st.session_state.clicked_street = clicked_name
            st.session_state.click_counter += 1
            st.rerun()

# 5.3 下方 3 个 Tab 面板（完好展示，彻底根治全局色彩覆盖带来的图表隐形 Bug）
tab1, tab2, tab3 = st.tabs(["🌊 潮汐曲线面板", "📈 POI-流量耦合", "🎯 热点接力流"])

# ---------- Tab 1: 潮汐曲线 ----------
with tab1:
    st.subheader("28 街道的 24h 流量曲线 (按潮汐类型着色)")
    if not has_tidal:
        st.info("潮汐标签仅在「核心 28 街道」视图下可用,请切换数据源。")
    else:
        street_options = core.sort_values("DISTRICT")["NAME"].tolist()
        default_idx = 0
        if st.session_state.clicked_street and st.session_state.clicked_street in street_options:
            default_idx = street_options.index(st.session_state.clicked_street)
        col_a, col_b = st.columns([3, 1])
        with col_a:
            sel_name = st.selectbox(
                "选择街道查看 24h 曲线",
                options=street_options,
                index=default_idx,
                key="tab1_sel_name_main",
            )
        with col_b:
            show_all = st.checkbox("叠加显示所有街道", value=False, key="tab1_show_all_main")

        fig, ax = plt.subplots(figsize=(13, 4.5))
        tidal_color = {
            "早峰型(工作地/CBD)": "#FF6B6B",
            "午间型(餐饮/办事)": "#FFD93D",
            "晚峰型(商圈/夜娱)": "#6BCB77",
            "深夜型(口岸/酒吧)": "#4D96FF",
            "未分类": "#888888",
        }
        core_plot = core.copy()
        for c in flow_cols_core:
            core_plot[c] = core_plot[c].astype(float)
        if show_all:
            for _, r in core_plot.iterrows():
                ax.plot(HOURS_ARR, r[flow_cols_core].values.astype(float),
                        color=tidal_color.get(r["tidal_type"], "#888"),
                        alpha=0.18, linewidth=0.8)
        sel = core_plot[core_plot["NAME"] == sel_name].iloc[0]
        sel_curve = sel[flow_cols_core].values.astype(float)
        ax.plot(HOURS_ARR, sel_curve, marker="o", linewidth=2.5,
                color=tidal_color.get(sel["tidal_type"], "#000"),
                label=f'{sel["NAME"]} ({sel["tidal_type"]})')
        ax.fill_between(HOURS_ARR, sel_curve, alpha=0.15,
                        color=tidal_color.get(sel["tidal_type"], "#000"))
        ax.set_xticks(HOURS)
        ax.set_xlabel("小时")
        ax.set_ylabel("车次")
        ax.set_title(f'{sel["NAME"]} · {sel["DISTRICT"]} · 24h 流量曲线')
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("##### 潮汐类型分布")
        type_counts = core["tidal_type"].value_counts().rename_axis("类型").reset_index(name="街道数")
        st.dataframe(type_counts, hide_index=True, width='stretch')

# ---------- Tab 2: POI 散点 ----------
with tab2:
    st.subheader("POI 数 vs 24h 流量峰值")
    if not has_poi:
        st.info("POI 散点仅在「核心 28 街道」视图下可用。")
    else:
        df_s = core_plot.copy()
        df_s["flow_max"]  = df_s[flow_cols_core].max(axis=1)
        df_s["flow_mean"] = df_s[flow_cols_core].mean(axis=1)
        df_s["flow_total"]= df_s[flow_cols_core].sum(axis=1)
        df_s["cv"]        = df_s[flow_cols_core].std(axis=1) / df_s["flow_mean"].replace(0, np.nan)

        col_a, col_b = st.columns(2)
        with col_a:
            metric_x = st.radio("X 轴", ["poi_count", "flow_total", "cv"], index=0, horizontal=True,
                                label_visibility="collapsed", key="tab2_metric_x_main")
        with col_b:
            metric_y = st.radio("Y 轴", ["flow_max", "flow_mean", "flow_total"], index=0, horizontal=True,
                                label_visibility="collapsed", key="tab2_metric_y_main")

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = {"福田区": "tab:blue", "罗湖区": "tab:orange", "南山区": "tab:green"}
        for d in df_s["DISTRICT"].unique():
            sub = df_s[df_s["DISTRICT"] == d]
            ax.scatter(sub[metric_x], sub[metric_y], s=80, alpha=0.75,
                       color=colors.get(d, "gray"), label=d,
                       edgecolors="k", linewidth=0.5)
        for _, r in df_s.iterrows():
            ax.annotate(r["NAME"], (r[metric_x], r[metric_y]),
                       xytext=(4, 4), textcoords="offset points",
                       fontsize=8, alpha=0.85)
        if df_s[metric_x].std() > 0:
            coef = np.polyfit(df_s[metric_x], df_s[metric_y], 1)
            xs = np.linspace(df_s[metric_x].min(), df_s[metric_x].max(), 50)
            ax.plot(xs, np.polyval(coef, xs), "r--",
                    label=f"线性拟合 y={coef[0]:.2f}x+{coef[1]:.1f}")
        ax.set_xlabel(metric_x); ax.set_ylabel(metric_y)
        ax.set_title(f"{metric_x} vs {metric_y}")
        ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        from scipy.stats import pearsonr, spearmanr
        r_p, p_p = pearsonr(df_s["poi_count"], df_s["flow_max"])
        r_s, p_s = spearmanr(df_s["poi_count"], df_s["flow_max"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pearson r (POI vs peak)", f"{r_p:.3f}")
        c2.metric("p-value", f"{p_p:.4f}")
        c3.metric("Spearman ρ", f"{r_s:.3f}")
        c4.metric("p-value", f"{p_s:.4f}")

# ---------- Tab 3: 接力流 ----------
with tab3:
    st.subheader("24h 热点接力(top-8 街道)")
    st.caption("h 时段的 top-8 街道,到 h+1 时段被替换,记为 enter/leave;持续在榜记为 stay。")
    sel_h = st.slider("查看从该小时起,到 23 时的接力", 0, 22, 8, key="tab3_sel_h_main")
    sub_relay = relay[relay["h"] == sel_h].copy()
    enter_n = (sub_relay["kind"] == "enter").sum()
    leave_n = (sub_relay["kind"] == "leave").sum()
    stay_n  = (sub_relay["kind"] == "stay").sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"h={sel_h:02d}→{sel_h+1:02d}", "切换边")
    c2.metric("进入 top", int(enter_n))
    c3.metric("离开 top", int(leave_n))
    c4.metric("继续占位", int(stay_n))
    st.dataframe(
        sub_relay[["from", "to", "kind"]].reset_index(drop=True),
        width='stretch', hide_index=True,
    )
    with st.expander("📋 查看完整 24h 接力表 (252 条)"):
        st.dataframe(relay, width='stretch', hide_index=True)

# ============================================================
# 7. 页脚
# ============================================================
st.markdown("---")
st.markdown(
    "**技术栈**:Python 3.12 · GeoPandas · Folium · Streamlit · scikit-learn  \n"
    "**数据源**:final-exp/data/(同学聚合 + 阶段②③ 产出)  \n"
    "**作者**:期末大作业 · 城市热点发现与动态演化"
)
