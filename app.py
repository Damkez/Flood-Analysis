import streamlit as st
import ee
import folium
from streamlit_folium import st_folium

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Analysis Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    section[data-testid="stSidebar"] { background-color: #1A1F2E; }
    .metric-card {
        background: linear-gradient(135deg, #1A1F2E, #2A3050);
        border: 1px solid #00CFFF33;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00CFFF; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    h1, h2, h3 { color: #FAFAFA !important; }
</style>
""", unsafe_allow_html=True)


# ── Earth Engine Initialization ────────────────────────────────────────────────
@st.cache_resource
def init_ee():
    """Initialize Earth Engine from Streamlit Secrets (handles both OAuth2 and Service Account)."""
    import json

    token_raw = st.secrets.get("EARTHENGINE_TOKEN", None)
    if not token_raw:
        st.error("No EARTHENGINE_TOKEN found in Streamlit Secrets.")
        return False

    try:
        creds_dict = json.loads(token_raw) if isinstance(token_raw, str) else dict(token_raw)
    except json.JSONDecodeError as e:
        st.error(f"Could not parse EARTHENGINE_TOKEN as JSON: {e}")
        return False

    try:
        # Service account: has 'client_email' and 'private_key'
        if "client_email" in creds_dict and "private_key" in creds_dict:
            import google.oauth2.service_account as sa
            scopes = ["https://www.googleapis.com/auth/earthengine"]
            credentials = sa.Credentials.from_service_account_info(creds_dict, scopes=scopes)
            ee.Initialize(credentials)

        # OAuth2 user credentials: has 'refresh_token' and 'client_id'
        elif "refresh_token" in creds_dict:
            import google.oauth2.credentials
            from google.auth.transport.requests import Request

            # Ensure token_uri exists
            if "token_uri" not in creds_dict:
                creds_dict["token_uri"] = "https://oauth2.googleapis.com/token"

            credentials = google.oauth2.credentials.Credentials(
                token=creds_dict.get("token"),
                refresh_token=creds_dict["refresh_token"],
                token_uri=creds_dict["token_uri"],
                client_id=creds_dict["client_id"],
                client_secret=creds_dict["client_secret"],
                scopes=creds_dict.get("scopes", ["https://www.googleapis.com/auth/earthengine"]),
            )
            # Refresh the token if it's expired
            if credentials.expired or not credentials.valid:
                credentials.refresh(Request())

            ee.Initialize(credentials)

        else:
            st.error("EARTHENGINE_TOKEN is not a recognised credential format.")
            return False

        return True

    except Exception as e:
        st.error(f"Earth Engine init failed: {e}")
        return False

ee_ready = init_ee()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Flood Mapping Dashboard")
    st.caption("Powered by Sentinel-1 SAR · Google Earth Engine")
    st.markdown("---")

    st.markdown("""
    **About**  
    This dashboard maps flood extents using **Synthetic Aperture Radar (SAR)** — 
    which penetrates clouds to detect floods even during active storms.

    **Case study:** 2022 Pakistan Monsoon Floods, Sindh Province
    """)

    st.markdown("---")
    st.markdown("### 🗂️ Map Layers")
    show_pre   = st.checkbox("Pre-Flood SAR",  value=False)
    show_post  = st.checkbox("Post-Flood SAR", value=False)
    show_flood = st.checkbox("Flood Extent",   value=True)
    show_perm  = st.checkbox("Permanent Water (JRC)", value=False)

    st.markdown("---")
    st.caption("Data: ESA Copernicus Sentinel-1 · JRC Global Surface Water")


# ── EE Layer Helpers ───────────────────────────────────────────────────────────
def get_tile_url(ee_image, vis_params):
    """Return an XYZ tile URL for an EE image (no geemap required)."""
    map_id = ee_image.getMapId(vis_params)
    return map_id["tile_fetcher"].url_format


# ── Data Processing ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Processing SAR imagery…")
def compute_flood_layers():
    AOI = ee.Geometry.Polygon([[[67.5, 25.5],[67.5, 27.5],[69.5, 27.5],[69.5, 25.5]]])

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(AOI)
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .select("VH"))

    pre_vh  = s1.filterDate("2022-05-01", "2022-06-30").mosaic().clip(AOI)
    post_vh = s1.filterDate("2022-08-15", "2022-09-15").mosaic().clip(AOI)

    r = 50  # speckle filter radius (m)
    pre_sm  = pre_vh.focal_mean(r, "circle", "meters")
    post_sm = post_vh.focal_mean(r, "circle", "meters")

    diff = post_sm.subtract(pre_sm)
    flooded_raw = diff.lt(-3).And(post_sm.lt(-16))

    perm_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("seasonality").gte(10)
    flood_ext  = flooded_raw.where(perm_water, 0).updateMask(flooded_raw.where(perm_water, 0))

    # Area stats (500m scale for speed)
    area_img = flood_ext.multiply(ee.Image.pixelArea()).divide(10000)
    stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=AOI, scale=500,
        maxPixels=1e9, bestEffort=True
    ).getInfo()
    flooded_ha = stats.get("VH", 0) or 0
    aoi_ha = AOI.area(maxError=1).getInfo() / 10000

    return pre_sm, post_sm, flood_ext, perm_water, flooded_ha, aoi_ha


# ── Main Layout ────────────────────────────────────────────────────────────────
st.markdown("# 🌊 Flood Analysis Dashboard — 2022 Pakistan Floods")
st.markdown("Sentinel-1 SAR backscatter change detection · Sindh Province, Pakistan")
st.markdown("---")

col_map, col_stats = st.columns([3, 1])

if not ee_ready:
    st.warning("Earth Engine is not initialised. Add EARTHENGINE_TOKEN to Streamlit Secrets.")
else:
    with st.spinner("Loading satellite data…"):
        pre_sm, post_sm, flood_ext, perm_water, flooded_ha, aoi_ha = compute_flood_layers()

    pct = (flooded_ha / aoi_ha * 100) if aoi_ha else 0
    flooded_km2 = flooded_ha / 100

    # ── Stats Column ─────────────────────────────────────────────────
    with col_stats:
        st.markdown("### 📊 Statistics")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{flooded_ha:,.0f}</div>
            <div class="metric-label">Flooded Area (Hectares)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{flooded_km2:,.0f}</div>
            <div class="metric-label">Flooded Area (km²)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{pct:.1f}%</div>
            <div class="metric-label">% of Study Area Affected</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📍 Event Details")
        st.info("""
**Event:** 2022 Pakistan Monsoon Floods

**Region:** Sindh Province

**Pre-Flood:** May – June 2022  
**Post-Flood:** Aug – Sep 2022

**Sensor:** Sentinel-1 GRD (VH pol.)

**Processing:** Speckle filter + backscatter change detection
        """)

    # ── Map Column ────────────────────────────────────────────────────
    with col_map:
        st.markdown("### 🗺️ Interactive Flood Map")

        m = folium.Map(location=[26.5, 68.5], zoom_start=8,
                       tiles="CartoDB dark_matter")

        sar_vis   = {"min": -25, "max": -5, "palette": ["000000", "ffffff"]}
        flood_vis = {"palette": ["00CFFF"]}
        perm_vis  = {"palette": ["0055FF"]}

        if show_pre:
            url = get_tile_url(pre_sm, sar_vis)
            folium.TileLayer(tiles=url, attr="Google Earth Engine",
                             name="Pre-Flood SAR", overlay=True).add_to(m)

        if show_post:
            url = get_tile_url(post_sm, sar_vis)
            folium.TileLayer(tiles=url, attr="Google Earth Engine",
                             name="Post-Flood SAR", overlay=True).add_to(m)

        if show_flood:
            url = get_tile_url(flood_ext, flood_vis)
            folium.TileLayer(tiles=url, attr="Google Earth Engine",
                             name="Flood Extent", overlay=True,
                             opacity=0.85).add_to(m)

        if show_perm:
            url = get_tile_url(perm_water.updateMask(perm_water), perm_vis)
            folium.TileLayer(tiles=url, attr="Google Earth Engine",
                             name="Permanent Water", overlay=True,
                             opacity=0.7).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=None, height=620, returned_objects=[])
