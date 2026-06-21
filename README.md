# 深圳市核心商圈 24h 出行潮汐与热点演化

本项目围绕深圳市出租车 GPS 与订单数据，分析核心商圈在 24 小时内的出行热点、潮汐特征与空间接力关系，并通过 Streamlit 构建交互式可视化大屏。

## 项目内容

- **数据体检**：检查街道聚合数据、原始 GPS/订单数据、街道边界与字段完整性。
- **OD 聚合**：将出租车订单与街道边界匹配，生成 74 个街道的小时级 pickup、dropoff、net 流量。
- **潮汐分析**：基于 24 小时流量曲线识别早高峰、晚高峰、双峰、平稳等潮汐类型。
- **热点接力**：分析不同时段热点街道之间的空间转移与接力边。
- **可视化大屏**：展示 24 小时街道热力图、潮汐类型、POI 相关性和接力网络。

## 目录结构

```text
.
├── app.py                     # Streamlit 可视化大屏入口
├── 01_data_audit.ipynb        # 数据可用性体检
├── 02_od_to_street_flow.ipynb # OD 到街道小时流量聚合
├── 03_tidal_analysis.ipynb    # 潮汐聚类、接力网络与 POI 相关分析
├── data/                      # 输入数据与分析产物
│   ├── GPS/                   # 出租车 GPS 轨迹 CSV
│   ├── business.csv           # 订单数据
│   ├── road.geojson           # 道路数据
│   ├── 深圳街道.geojson        # 深圳街道边界
│   ├── streets_24h_core.geojson
│   ├── street_hour_full.geojson
│   ├── tidal_clusters.csv
│   ├── relay_edges.csv
│   └── poi_corr.json
└── output/                    # 运行 notebook 后生成的中间结果/最终结果
```

## 数据说明

当前项目已将原 `final-exp/data/` 与 `exp6/data/` 数据合并到统一的 `data/` 目录下，因此 notebook 中使用相对路径读取数据：

```python
ROOT = os.getcwd()
FINAL = os.path.join(ROOT, 'data')
EXP6 = FINAL
OUT = os.path.join(ROOT, 'output')
```

主要数据文件：

| 文件 | 说明 |
|---|---|
| `data/streets_24h_core.geojson` | 核心 28 街道 24 小时流量与 POI 数据 |
| `data/street_hour_full.geojson` | 74 街道小时级 pickup/dropoff/net 流量 |
| `data/tidal_clusters.csv` | 街道潮汐类型聚类结果 |
| `data/relay_edges.csv` | 热点接力边结果 |
| `data/poi_corr.json` | POI 与流量相关性分析结果 |
| `data/GPS/` | 出租车 GPS 轨迹原始 CSV |
| `data/business.csv` | 出租车订单数据 |
| `data/深圳街道.geojson` | 深圳街道边界数据 |

## 运行环境

建议使用 Python 3.10+。主要依赖包括：

```bash
pip install numpy pandas geopandas matplotlib shapely scikit-learn scipy streamlit folium streamlit-folium
```

如果 `geopandas` 安装失败，建议使用 conda：

```bash
conda install -c conda-forge geopandas shapely pyproj fiona
pip install streamlit folium streamlit-folium scikit-learn scipy
```

## 使用方式

### 1. 运行 notebook

建议按顺序执行：

1. `01_data_audit.ipynb`：确认数据完整性与字段一致性。
2. `02_od_to_street_flow.ipynb`：生成 74 街道小时级流量数据。
3. `03_tidal_analysis.ipynb`：生成潮汐聚类、接力网络和 POI 相关性结果。

### 2. 启动可视化大屏

在项目根目录运行：

```bash
streamlit run app.py
```

启动后可在浏览器中查看：

- 24 小时街道流量热力图
- 起点流量、终点流量、净流入视角切换
- 核心 28 街道与全市 74 街道覆盖范围切换
- 潮汐类型、POI 相关性与热点接力分析

## 输出结果

notebook 和大屏主要依赖以下结果文件：

- `data/street_hour_full.geojson`
- `data/tidal_clusters.csv`
- `data/relay_edges.csv`
- `data/poi_corr.json`

如果重新运行分析流程，部分结果也会输出到 `output/` 目录。

## 注意事项

- 请从项目根目录运行 notebook 或 Streamlit，否则相对路径可能无法正确解析。
- `data/` 目录已经包含原始数据和分析产物，不需要再单独创建 `exp6/` 目录。
- Windows 环境下建议安装中文字体，如 `Microsoft YaHei` 或 `SimHei`，避免图表中文乱码。
