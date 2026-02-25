# Flood Mapping and Analysis Dashboard 🌊

This repository contains a full pipeline for visualizing and analyzing flood extents using Google Earth Engine and Sentinel-1 Synthetic Aperture Radar (SAR) imagery. Built with Python and Streamlit.

## Components

1. `flood_analysis.ipynb`: A Jupyter Notebook that details the methodology. It walks through Earth Engine authentication, data retrieval (Sentinel-1 GRD), speckle filtering, and backscatter thresholding to compute the actual flooded area in hectares.
2. `app.py`: An interactive, deployment-ready dashboard built in Streamlit and Geemap. It presents a map-centric view comparing Pre-Flood radar, Post-Flood radar, and the extracted Flood Extent.

## Running Locally

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   env\Scripts\activate  # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit App:
   ```bash
   streamlit run app.py
   ```
   > Note: Your browser will open a local server at `http://localhost:8501`. If it's your first time using Earth Engine on your machine, you might need to run `earthengine authenticate` in your terminal.

## Deploying to Streamlit Community Cloud (Free)

1. **GitHub Repository:** Push these files (`app.py`, `requirements.txt`, `README.md`, and the `ipynb`) to a public or private GitHub repository.
2. **Streamlit Connection:** 
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in (or create an account using your GitHub credentials).
   - Click **"New app"**.
3. **App Details:**
   - Select the repository you just created.
   - Branch: `main` (or `master`)
   - Main file path: `app.py`
4. **Earth Engine Credentials (Crucial!):** 
   Because Earth Engine requires an account, the cloud computer running your code needs your credentials.
   - Go to the **Advanced Settings** (or click the ⚙️ icon) during deployment.
   - In the **Secrets** box, you need to provide your Earth Engine service account key JSON or a short-lived token. (For personal use, see [Geemap deployment docs](https://geemap.org/notebooks/115_streamlit/) for generating an Earth Engine token for Streamlit secrets). 
5. **Deploy:** Click **"Deploy!"**. The build process might take a few minutes as it installs `geemap` and other dependencies.

---
**Author:** Damilola Akindele
/n
**Data Source:** Data provided by Copernicus EU / European Space Agency / Google Earth Engine.
