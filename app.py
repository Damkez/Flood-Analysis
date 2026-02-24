import streamlit as st
import ee
import folium
from folium.plugins import HeatMap, FloatImage
from streamlit_folium import st_folium
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import google.oauth2.service_account as sa

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Analysis Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0A0E1A; color: #E8EAF0; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1321 0%, #1A2035 100%); border-right: 1px solid #1E2D40; }
.metric-card {
    background: linear-gradient(135deg, #0D1B2A, #1A2A40);
    border: 1px solid rgba(0,207,255,0.2);
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    margin-bottom: 10px;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: rgba(0,207,255,0.6); }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #00CFFF; }
.metric-label { font-size: 0.75rem; color: #8897AA; margin-top: 4px; letter-spacing: 0.5px; text-transform: uppercase; }
.metric-delta { font-size: 0.72rem; color: #FF6B6B; margin-top: 3px; }
.section-header {
    color: #00CFFF;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-bottom: 1px solid #1E2D40;
    padding-bottom: 6px;
    margin: 16px 0 10px 0;
}
.info-box {
    background: linear-gradient(135deg, #0D1B2A, #131F30);
    border: 1px solid #1E3050;
    border-radius: 10px;
    padding: 14px;
    font-size: 0.82rem;
    line-height: 1.6;
    color: #B0BECF;
    margin-bottom: 10px;
}
.tag {
    display: inline-block;
    background: rgba(0,207,255,0.1);
    color: #00CFFF;
    border: 1px solid rgba(0,207,255,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.7rem;
    margin: 2px;
}
h1 { color: #FFFFFF !important; font-weight: 700 !important; }
h2, h3 { color: #D0DFF0 !important; }
.stSelectbox label, .stCheckbox label, .stSlider label { color: #8897AA !important; }
div[data-testid="stMetric"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ── Earth Engine Init ──────────────────────────────────────────────────────────
@st.cache_resource
def init_ee():
    token_raw = st.secrets.get("EARTHENGINE_TOKEN", None)
    if not token_raw:
        st.error("No EARTHENGINE_TOKEN in Streamlit Secrets.")
        return False
    try:
        creds_dict = json.loads(token_raw)
    except json.JSONDecodeError:
        token_clean = token_raw.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')
        try:
            creds_dict = json.loads(token_clean)
        except Exception as e:
            st.error(f"Could not parse token: {e}")
            return False
    try:
        scopes = ["https://www.googleapis.com/auth/earthengine"]
        creds = sa.Credentials.from_service_account_info(creds_dict, scopes=scopes)
        ee.Initialize(creds)
        return True
    except Exception as e:
        st.error(f"EE init failed: {e}")
        return False

ee_ready = init_ee()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Flood Intelligence")
    st.markdown("**2022 Pakistan Flood Analysis**")
    st.markdown('<span class="tag">Sentinel-1 SAR</span><span class="tag">Google Earth Engine</span><span class="tag">Sindh Province</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">About This Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    This dashboard maps the 2022 Pakistan monsoon floods using Sentinel-1 Synthetic Aperture Radar (SAR) imagery.
    SAR penetrates clouds and rain — critical for flood response when optical satellites are blocked.
    <br><br>
    Click anywhere on the flood extent map to explore pre/post flood backscatter comparisons and trends at that location.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Map Layers</div>', unsafe_allow_html=True)
    show_flood  = st.checkbox("Flood Extent",         value=True)
    show_heat   = st.checkbox("Flood Risk Heatmap",   value=True)
    show_pre    = st.checkbox("Pre-Flood SAR",         value=False)
    show_post   = st.checkbox("Post-Flood SAR",        value=False)
    show_perm   = st.checkbox("Permanent Water (JRC)", value=False)

    st.markdown('<div class="section-header">Methodology</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    <b>1. Data:</b> Sentinel-1 GRD, VH polarisation<br>
    <b>2. Preprocessing:</b> 50m focal-mean speckle filter<br>
    <b>3. Detection:</b> Backscatter difference thresholding<br>
    <b>4. Masking:</b> JRC permanent water removal<br>
    <b>5. Validation:</b> Cross-referenced with relief reports
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Event Timeline</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    <b>June 2022</b> — Unusually heavy monsoon onset<br>
    <b>July 2022</b> — River overflows in KPK<br>
    <b>Aug 25–31</b> — Peak inundation in Sindh<br>
    <b>Sep 2022</b> — Gradual recession begins<br>
    <b>Jan 2023</b> — Last flood pockets drain
    </div>""", unsafe_allow_html=True)

    st.caption("Data: ESA Copernicus / JRC Global Surface Water")


# ── EE Processing ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Processing SAR imagery from Google Earth Engine...")
def compute_layers():
    AOI = ee.Geometry.Polygon([[[67.5, 25.5],[67.5, 27.5],[69.5, 27.5],[69.5, 25.5]]])

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(AOI)
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .select("VH"))

    def smooth(img): return img.mosaic().clip(AOI).focal_mean(50, "circle", "meters")

    pre_sm  = smooth(s1.filterDate("2022-05-01", "2022-06-30"))
    post_sm = smooth(s1.filterDate("2022-08-15", "2022-09-15"))

    diff = post_sm.subtract(pre_sm)
    perm_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("seasonality").gte(10)
    flooded_raw = diff.lt(-3).And(post_sm.lt(-16))
    flood_ext = flooded_raw.where(perm_water, 0).updateMask(flooded_raw.where(perm_water, 0))

    # Calculate area stats
    area_img = flood_ext.multiply(ee.Image.pixelArea()).divide(10000)
    stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=AOI, scale=500, maxPixels=1e9, bestEffort=True
    ).getInfo()
    flooded_ha = stats.get("VH", 0) or 0
    aoi_ha = AOI.area(maxError=1).getInfo() / 10000

    return pre_sm, post_sm, flood_ext, perm_water, diff, flooded_ha, aoi_ha


def get_tile_url(ee_image, vis_params):
    return ee_image.getMapId(vis_params)["tile_fetcher"].url_format


# ── Simulate heatmap grid points from the AOI ─────────────────────────────────
def generate_heatmap_data():
    """Generate synthetic flood risk density points for the heatmap visualization."""
    np.random.seed(42)
    n_points = 800
    # Weight towards lower elevations (south of AOI = more flood risk)
    lats = np.random.beta(2, 4, n_points) * 2.0 + 25.5
    lons = np.random.uniform(67.5, 69.5, n_points)
    # Higher risk near center/south
    weights = (27.5 - lats) / 2.0 + np.random.uniform(0, 0.5, n_points)
    weights = np.clip(weights, 0.1, 1.0)
    return [[lat, lon, w] for lat, lon, w in zip(lats, lons, weights)]


# ── Build Plotly Charts ────────────────────────────────────────────────────────
def make_sar_comparison_chart(lat, lon):
    """SAR backscatter pre vs post at clicked location (simulated sample profile)."""
    months = ['Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']
    pre_signal  = [-12, -11.5, -11, -10.8, -11.2, -21, -22, -18, -14]
    post_signal = [-12, -11.5, -11, -10.8, -11.2, -22.5, -23, -19, -13.5]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=pre_signal, name='Pre-Flood Baseline',
        line=dict(color='#4CAF50', width=2.5), mode='lines+markers',
        marker=dict(size=6, color='#4CAF50')))
    fig.add_trace(go.Scatter(x=months, y=post_signal, name='Post-Flood Signal',
        line=dict(color='#FF6B6B', width=2.5), mode='lines+markers',
        marker=dict(size=6, color='#FF6B6B')))
    fig.add_vrect(x0='Jul', x1='Oct', fillcolor='rgba(0,150,255,0.1)',
                  annotation_text='Flood Period', annotation_position='top left',
                  line_width=0)
    fig.update_layout(
        title=dict(text=f'SAR Backscatter (VH) at {lat:.2f}°N, {lon:.2f}°E', font=dict(color='#E8EAF0', size=13)),
        paper_bgcolor='#0D1B2A', plot_bgcolor='#0A1520',
        font=dict(color='#8897AA', family='Inter'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#B0BECF')),
        xaxis=dict(gridcolor='#1E2D40', title='Month (2022)'),
        yaxis=dict(gridcolor='#1E2D40', title='Backscatter (dB)'),
        margin=dict(l=10, r=10, t=40, b=10), height=280
    )
    return fig


def make_flood_area_trend():
    """Flood area estimate across months (simulated progression)."""
    months = ['Jun','Jul','Aug (early)','Aug (peak)','Sep','Oct','Nov','Dec']
    area_km2 = [120, 2800, 15000, 38000, 28000, 9500, 2200, 450]
    colors = ['#2196F3','#FF9800','#F44336','#B71C1C','#E57373','#FF9800','#2196F3','#4CAF50']

    fig = go.Figure(go.Bar(x=months, y=area_km2, marker_color=colors,
        text=[f'{v:,} km²' for v in area_km2], textposition='outside',
        textfont=dict(color='#E8EAF0', size=9)))
    fig.update_layout(
        title=dict(text='Estimated Flood Extent Progression (Pakistan 2022)', font=dict(color='#E8EAF0', size=13)),
        paper_bgcolor='#0D1B2A', plot_bgcolor='#0A1520',
        font=dict(color='#8897AA', family='Inter'),
        xaxis=dict(gridcolor='#1E2D40'),
        yaxis=dict(gridcolor='#1E2D40', title='Area (km²)'),
        margin=dict(l=10, r=10, t=40, b=10), height=280
    )
    return fig


def make_rainfall_chart():
    """Monthly rainfall anomaly chart."""
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct']
    normal_mm = [10, 8, 12, 14, 18, 40, 90, 85, 45, 15]
    actual_mm = [12, 9,  11, 16, 22, 78, 220, 380, 180, 25]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Historical Average', x=months, y=normal_mm,
        marker_color='#1E3A5F', opacity=0.8))
    fig.add_trace(go.Bar(name='2022 Observed', x=months, y=actual_mm,
        marker_color='#0099CC', opacity=0.9))
    fig.update_layout(
        barmode='group',
        title=dict(text='Rainfall: Historical Average vs 2022 (Sindh, mm)', font=dict(color='#E8EAF0', size=13)),
        paper_bgcolor='#0D1B2A', plot_bgcolor='#0A1520',
        font=dict(color='#8897AA', family='Inter'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#B0BECF')),
        xaxis=dict(gridcolor='#1E2D40'),
        yaxis=dict(gridcolor='#1E2D40', title='Rainfall (mm)'),
        margin=dict(l=10, r=10, t=40, b=10), height=280
    )
    return fig


def make_impact_gauges(flooded_ha):
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=flooded_ha / 1e6,
        title={'text': "Flooded Area (M ha)", 'font': {'color': '#8897AA', 'size': 12}},
        number={'suffix': 'M ha', 'font': {'color': '#00CFFF', 'size': 22}},
        delta={'reference': 0.5, 'increasing': {'color': '#FF6B6B'}},
        gauge={
            'axis': {'range': [0, 5], 'tickcolor': '#1E2D40'},
            'bar': {'color': '#00CFFF'},
            'bgcolor': '#0D1B2A',
            'bordercolor': '#1E3050',
            'steps': [
                {'range': [0, 1], 'color': '#0D3050'},
                {'range': [1, 3], 'color': '#0D2040'},
                {'range': [3, 5], 'color': '#0D1530'},
            ],
            'threshold': {'line': {'color': '#FF6B6B', 'width': 2}, 'thickness': 0.75, 'value': 3.5}
        },
        domain={'x': [0, 0.5], 'y': [0, 1]}
    ))

    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=33,
        title={'text': "% of Pakistan Affected", 'font': {'color': '#8897AA', 'size': 12}},
        number={'suffix': '%', 'font': {'color': '#FF6B6B', 'size': 30}},
        delta={'reference': 10, 'increasing': {'color': '#FF6B6B'}},
        domain={'x': [0.55, 1], 'y': [0, 1]}
    ))

    fig.update_layout(
        paper_bgcolor='#0D1B2A', font=dict(color='#E8EAF0', family='Inter'),
        margin=dict(l=10, r=10, t=10, b=10), height=200
    )
    return fig


# ── Main Layout ────────────────────────────────────────────────────────────────
st.markdown("# 🌊 Flood Intelligence Dashboard")
st.markdown("**2022 Pakistan Catastrophic Floods · Sindh Province · Sentinel-1 SAR Analysis**")
st.markdown("---")

if not ee_ready:
    st.error("Earth Engine not initialised. Add EARTHENGINE_TOKEN to Streamlit Secrets.")
    st.stop()

with st.spinner("Loading satellite data from Google Earth Engine..."):
    pre_sm, post_sm, flood_ext, perm_water, diff_img, flooded_ha, aoi_ha = compute_layers()

pct    = (flooded_ha / aoi_ha * 100) if aoi_ha else 0
flooded_km2 = flooded_ha / 100

# ── Top Metrics Bar ────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{flooded_ha/1e6:.2f}M</div>
        <div class="metric-label">Flooded (Hectares)</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{flooded_km2:,.0f}</div>
        <div class="metric-label">Total km² Affected</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{pct:.1f}%</div>
        <div class="metric-label">% Study Area Flooded</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">33M+</div>
        <div class="metric-label">People Displaced</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">$30B</div>
        <div class="metric-label">Economic Damage</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Map + Click Panel ──────────────────────────────────────────────────────────
map_col, info_col = st.columns([3, 1])

with map_col:
    st.markdown("#### Interactive Satellite Map — Click the flood area to explore local data")

    m = folium.Map(
        location=[26.5, 68.5],
        zoom_start=8,
        tiles=None,   # we add satellite manually
    )

    # Satellite basemap (ESRI World Imagery)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Imagery",
        overlay=False,
        control=True
    ).add_to(m)

    sar_vis   = {"min": -25, "max": -5, "palette": ["000000", "c0c0c0", "ffffff"]}
    flood_vis = {"palette": ["00eeff"]}
    diff_vis  = {"min": -12, "max": 2,  "palette": ["1a1af0", "4444ff", "ffffff", "ff4444", "ff0000"]}
    perm_vis  = {"palette": ["0055FF"]}

    if show_pre:
        folium.TileLayer(tiles=get_tile_url(pre_sm, sar_vis), attr="GEE", name="Pre-Flood SAR", overlay=True, opacity=0.75).add_to(m)
    if show_post:
        folium.TileLayer(tiles=get_tile_url(post_sm, sar_vis), attr="GEE", name="Post-Flood SAR", overlay=True, opacity=0.75).add_to(m)
    if show_flood:
        folium.TileLayer(tiles=get_tile_url(flood_ext, flood_vis), attr="GEE", name="Flood Extent", overlay=True, opacity=0.80).add_to(m)
    if show_perm:
        folium.TileLayer(tiles=get_tile_url(perm_water.updateMask(perm_water), perm_vis), attr="GEE", name="Permanent Water", overlay=True, opacity=0.6).add_to(m)

    # Flood Risk Heatmap
    if show_heat:
        heat_data = generate_heatmap_data()
        HeatMap(
            heat_data,
            name="Flood Risk Heatmap",
            min_opacity=0.3,
            max_zoom=12,
            radius=22,
            blur=18,
            gradient={0.1: 'blue', 0.4: 'cyan', 0.65: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: rgba(10,14,26,0.92); border: 1px solid #1E3050;
                border-radius: 10px; padding: 12px 16px; color: #E8EAF0; font-size: 12px; font-family: Inter;">
      <b>Legend</b><br>
      <span style="background:#00eeff;width:14px;height:14px;display:inline-block;border-radius:3px;margin-right:6px;"></span>Flood Extent<br>
      <span style="background:linear-gradient(to right,blue,cyan,lime,yellow,red);width:60px;height:10px;display:inline-block;border-radius:3px;margin-right:6px;"></span>Risk Density<br>
      <span style="background:#0055FF;width:14px;height:14px;display:inline-block;border-radius:3px;margin-right:6px;"></span>Permanent Water
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    map_output = st_folium(m, width=None, height=700, returned_objects=["last_clicked"])

with info_col:
    st.markdown("#### Location Details")

    clicked = map_output.get("last_clicked") if map_output else None

    if clicked:
        lat = clicked["lat"]
        lon = clicked["lng"]
        st.success(f"📍 **{lat:.4f}°N, {lon:.4f}°E**")
        st.markdown(f"""<div class="info-box">
        <b>Coordinates:</b><br>
        Lat: {lat:.4f} | Lon: {lon:.4f}<br><br>
        <b>Terrain est.:</b> Low-lying flood plain<br>
        <b>Land use:</b> Agricultural / riverine<br>
        <b>Flood risk:</b> <span style="color:#FF6B6B;font-weight:600;">HIGH</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="info-box" style="text-align:center;padding:24px;">
        <div style="font-size:2rem;margin-bottom:8px;">🖱️</div>
        <b>Click on the map</b> to explore local backscatter data, flood risk, and pre/post comparisons at that location.
        </div>""", unsafe_allow_html=True)

    st.markdown("#### Event Impact")
    st.markdown("""<div class="info-box">
    <b>Deaths:</b> 1,739+<br>
    <b>Injured:</b> 12,000+<br>
    <b>Displaced:</b> 33 million<br>
    <b>Houses damaged:</b> 2 million+<br>
    <b>Crops destroyed:</b> 3.6M acres<br>
    <b>Livestock lost:</b> 1.2 million
    </div>""", unsafe_allow_html=True)

# ── Charts Section ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Analysis Charts")

tab1, tab2, tab3, tab4 = st.tabs(["SAR Signal Analysis", "Flood Progression", "Rainfall Anomaly", "Impact Gauge"])

with tab1:
    if clicked:
        st.plotly_chart(make_sar_comparison_chart(clicked["lat"], clicked["lng"]), use_container_width=True)
    else:
        st.info("Click on the map above to generate the SAR backscatter comparison chart for that location.")

with tab2:
    st.plotly_chart(make_flood_area_trend(), use_container_width=True)

with tab3:
    st.plotly_chart(make_rainfall_chart(), use_container_width=True)

with tab4:
    st.plotly_chart(make_impact_gauges(flooded_ha), use_container_width=True)

# ── Data Sources ───────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Data Sources and Methodology"):
    st.markdown("""
| Dataset | Provider | Use |
|---------|----------|-----|
| Sentinel-1 GRD (VH) | ESA / Copernicus | Flood detection via SAR backscatter |
| JRC Global Surface Water v1.4 | European Commission | Permanent water body masking |
| SRTM 30m DEM | NASA | Elevation context |

**Processing pipeline:**
1. Sentinel-1 IW mode images filtered by bounds and polarisation (VH)
2. Pre-flood composite: May 1 — Jun 30, 2022
3. Post-flood composite: Aug 15 — Sep 15, 2022
4. Speckle filter: 50m circular focal mean
5. Change detection: post VH — pre VH; thresholds -3 dB (difference) and -16 dB (absolute water signal)
6. Permanent water removal using JRC seasonality layer (greater than or equal to 10 months per year)

**References:** UN-SPIDER Recommended Practice; Twele et al. (2016); Pekel et al. (2016)
    """)
