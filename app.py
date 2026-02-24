import streamlit as st
import ee
import geemap.foliumap as geemap

# --- Page Config ---
st.set_page_config(
    page_title="Flood Analysis Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Earth Engine ---
# Streamlit Community Cloud uses secrets for authentication
@st.cache_resource
def ee_authenticate(token_name="EARTHENGINE_TOKEN"):
    try:
        ee.Initialize(project='ee-your-project-id') # Update with a project id if needed
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()

ee_authenticate()


# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Satellite_icon.svg/1024px-Satellite_icon.svg.png", width=100)
    st.title("🌊 Flood Mapping Analysis")
    st.markdown("---")
    
    st.markdown(
        """
        **About this Dashboard**
        This application visualizes flood extents using Synthetic Aperture Radar (SAR) imagery from the Sentinel-1 satellite.
        
        SAR can penetrate clouds and rain, making it the industry standard for mapping floods during persistent weather events.
        """
    )
    
    st.markdown("### Controls")
    
    # Example event selection
    event = st.selectbox(
        "Select Event",
        ("2022 Pakistan Floods (Sindh)", "Custom Area")
    )
    
    # Default parameters for Pakistan Floods
    default_pre_start = '2022-05-01'
    default_pre_end = '2022-06-30'
    default_post_start = '2022-08-15'
    default_post_end = '2022-09-15'
    
    # Map layers toggles
    st.markdown("### Layers to Display")
    show_pre = st.checkbox("Show Pre-Flood SAR", value=False)
    show_post = st.checkbox("Show Post-Flood SAR", value=False)
    show_flood = st.checkbox("Show Flood Extent", value=True)
    
    st.markdown("---")
    st.info("Data Source: Copernicus EU / European Space Agency / Google Earth Engine")

# --- Processing Logic ---

# Define AOI for Pakistan Event
aoi = ee.Geometry.Polygon(
    [[[67.5, 25.5],
      [67.5, 27.5],
      [69.5, 27.5],
      [69.5, 25.5]]]
)

center_lat, center_lon = 26.5, 68.5

@st.cache_data
def process_flood_data(pre_s, pre_e, post_s, post_e):
    # Retrieve imagery
    collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW'))

    pre_flood = collection.filterDate(pre_s, pre_e).mosaic().clip(aoi)
    post_flood = collection.filterDate(post_s, post_e).mosaic().clip(aoi)

    pre_vh = pre_flood.select('VH')
    post_vh = post_flood.select('VH')
    
    # Speckle filter
    smoothing_radius = 50 
    pre_smoothed = pre_vh.focal_mean(smoothing_radius, 'circle', 'meters')
    post_smoothed = post_vh.focal_mean(smoothing_radius, 'circle', 'meters')

    # Difference and threshold
    difference = post_smoothed.subtract(pre_smoothed)
    flood_threshold = -3
    water_threshold = -16

    flooded = difference.lt(flood_threshold).And(post_smoothed.lt(water_threshold))
    flood_mask = flooded.updateMask(flooded)
    
    return pre_smoothed, post_smoothed, flood_mask


pre_map, post_map, flood_layer = process_flood_data(
    default_pre_start, default_pre_end, default_post_start, default_post_end
)


# --- Main Layout ---
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Interactive Map View")
    Map = geemap.Map(center=[center_lat, center_lon], zoom=8, add_google_map=False)
    Map.add_basemap('SATELLITE')
    
    if show_pre:
        Map.addLayer(pre_map, {'min': -25, 'max': -5}, 'Pre-flood SAR', True)
    if show_post:
        Map.addLayer(post_map, {'min': -25, 'max': -5}, 'Post-flood SAR', True)
    if show_flood:
        Map.addLayer(flood_layer, {'palette': ['00FFFF']}, 'Flood Extent (Cyan)')

    Map.to_streamlit(height=650)


with col2:
    st.markdown("### Statistics (Estimated)")
    
    # The calculation on EE is slow and can timeout if done across a huge area in the UI directly.
    # So we display an estimated static value or calculate a very downsampled version.
    # Let's do a fast reduced calculation
    with st.spinner("Calculating affected area..."):
        try:
            # Downsampled scale for faster UI performance
            area_img = flood_layer.multiply(ee.Image.pixelArea()).divide(10000)
            stats = area_img.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi,
                scale=500, # 500m scale for fast approximation
                maxPixels=1e9
            ).getInfo()
            
            calculated_area = stats.get('VH', 0)
            st.metric(label="Estimated Flooded Area", value=f"{calculated_area:,.2f} ha")
        except Exception as e:
            st.metric(label="Estimated Flooded Area", value="Calculation Error")
            st.error(f"Error: {e}")
            
    st.markdown(
        """
        > **Note:** The area calculation is an approximation running at a downsampled resolution (500m) 
        > to prevent timeouts within the interactive dashboard. See the Jupyter Notebook for full-resolution calculations.
        """
    )
