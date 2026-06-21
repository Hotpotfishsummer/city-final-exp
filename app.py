"""
深圳市核心商圈 24h 出行潮汐与热点演化大屏
========================================
- 入口: `streamlit run app.py`
- 数据: 与 app.py 同级的 data/ 目录
  - data/streets_24h_core.geojson   (28 街道 × 24h × POI, 同学聚合)
  - data/street_hour_full.geojson   (74 街道 × 24h pickup/dropoff/net, 阶段② 产出)
  - data/tidal_clusters.csv         (28 街道潮汐聚类标签, 阶段③ 产出)
  - data/relay_edges.csv            (24h 接力边, 阶段③ 产出)
  - data/poi_corr.json              (POI-流量相关系数, 阶段③ 产出)

设计:
- 左侧栏:时间轴 + 视图模式 + 区选择
- 主体上半:暗色 Choropleth 热力图 (主图)
- 主体下半: 3 个分页指标卡 (潮汐面板 / POI 散点 / 接力流)
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

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
# 4. 主图:暗色 Folium
# ============================================================
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

# 4.2 大区边界 (用核心 28 街道的 district_gdf 做参照)
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
    # 找出当前小时 top-8
    cur_top = core.nlargest(8, f"flow_{cur_hour}")["NAME"].tolist()
    next_top = core.nlargest(8, f"flow_{next_hour}")["NAME"].tolist()
    # 街道几何字典
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

# 4.5 Tooltip 浮窗
if has_poi:
    tip_fields = ["DISTRICT", "NAME", "poi_count", view_col, "tidal_type"]
    tip_aliases = ["🏢 大区:", "📍 街道:", "🏪 POI 数:", f"🔥 {HOURS_LBL[selected_hour]} {view_mode}:", "🌊 潮汐类型:"]
else:
    tip_fields = ["DISTRICT", "NAME", view_col]
    tip_aliases = ["🏢 大区:", "📍 街道:", f"🔥 {HOURS_LBL[selected_hour]} {view_mode}:"]

tip = folium.features.GeoJson(
    gdf,
    style_function=lambda x: {"fillColor": "#00000000", "color": "#00000000"},
    highlight_function=lambda x: {"weight": 3, "color": "white", "fillOpacity": 0.1},
    tooltip=folium.features.GeoJsonTooltip(
        fields=tip_fields, aliases=tip_aliases,
        style=("background-color:rgba(20,20,20,0.9);color:white;"
               "border-radius:4px;padding:10px;font-size:13px;"),
    ),
)
m.add_child(tip)

# ============================================================
# 5. 渲染主图
# ============================================================
st_folium(m, width="100%", height=620, returned_objects=[])

# ============================================================
# 6. 下方 3 个分页 (Tabs): 潮汐面板 / POI 散点 / 接力流
# ============================================================
tab1, tab2, tab3 = st.tabs(["🌊 潮汐曲线面板", "📈 POI-流量耦合", "🎯 热点接力流"])

# ---------- Tab 1: 潮汐曲线 ----------
with tab1:
    st.subheader(f"28 街道的 24h 流量曲线 (按潮汐类型着色)")
    if not has_tidal:
        st.info("潮汐标签仅在「核心 28 街道」视图下可用,请切换数据源。")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            # 选街道
            sel_name = st.selectbox(
                "选择街道查看 24h 曲线",
                options=core.sort_values("DISTRICT")["NAME"].tolist(),
                index=0,
            )
        with col_b:
            show_all = st.checkbox("叠加显示所有街道", value=False)

        # 画图
        fig, ax = plt.subplots(figsize=(13, 4.5))
        # 配色:按潮汐类型
        tidal_color = {
            "早峰型(工作地/CBD)": "#FF6B6B",
            "午间型(餐饮/办事)": "#FFD93D",
            "晚峰型(商圈/夜娱)": "#6BCB77",
            "深夜型(口岸/酒吧)": "#4D96FF",
            "未分类": "#888888",
        }
        # ⚠️ 关键:streets_24h_core 的 flow_h 是 int (但被 GeoJSON 读成 int64),
        # 为安全起见,所有 .values 显式转 float
        core_plot = core.copy()
        for c in flow_cols_core:
            core_plot[c] = core_plot[c].astype(float)
        if show_all:
            for _, r in core_plot.iterrows():
                ax.plot(HOURS_ARR, r[flow_cols_core].values.astype(float),
                        color=tidal_color.get(r["tidal_type"], "#888"),
                        alpha=0.18, linewidth=0.8)
        # 选中的街道高亮
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

        # 潮汐类型分布
        st.markdown("##### 潮汐类型分布")
        type_counts = core["tidal_type"].value_counts().rename_axis("类型").reset_index(name="街道数")
        st.dataframe(type_counts, hide_index=True, use_container_width=True)

# ---------- Tab 2: POI 散点 ----------
with tab2:
    st.subheader("POI 数 vs 24h 流量峰值")
    if not has_poi:
        st.info("POI 散点仅在「核心 28 街道」视图下可用。")
    else:
        # 准备散点数据
        df_s = core_plot.copy()
        df_s["flow_max"]  = df_s[flow_cols_core].max(axis=1)
        df_s["flow_mean"] = df_s[flow_cols_core].mean(axis=1)
        df_s["flow_total"]= df_s[flow_cols_core].sum(axis=1)
        df_s["cv"]        = df_s[flow_cols_core].std(axis=1) / df_s["flow_mean"].replace(0, np.nan)

        col_a, col_b = st.columns(2)
        with col_a:
            metric_x = st.radio("X 轴", ["poi_count", "flow_total", "cv"], index=0, horizontal=True,
                                label_visibility="collapsed")
        with col_b:
            metric_y = st.radio("Y 轴", ["flow_max", "flow_mean", "flow_total"], index=0, horizontal=True,
                                label_visibility="collapsed")

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
        # 趋势线
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

        # 相关系数
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

    # 选小时查看
    sel_h = st.slider("查看从该小时起,到 23 时的接力", 0, 22, 8)

    # 选中的接力子集
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
        use_container_width=True, hide_index=True,
    )

    # 整体 24h 接力表
    with st.expander("📋 查看完整 24h 接力表 (252 条)"):
        st.dataframe(relay, use_container_width=True, hide_index=True)

# ============================================================
# 7. 页脚
# ============================================================
st.markdown("---")
st.markdown(
    "**技术栈**:Python 3.12 · GeoPandas · Folium · Streamlit · scikit-learn  \n"
    "**数据源**:final-exp/data/(同学聚合 + 阶段②③ 产出)  \n"
    "**作者**:期末大作业 · 城市热点发现与动态演化"
)
