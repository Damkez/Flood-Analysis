import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Flood Analysis using Sentinel-1 SAR (Google Earth Engine)\n",
    "\n",
    "This notebook demonstrates how to process Sentinel-1 Synthetic Aperture Radar (SAR) imagery to detect flood extents. We will use the 2022 Pakistan Floods as a case study."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import ee\n",
    "import geemap\n",
    "\n",
    "# Authenticate and Initialize Earth Engine\n",
    "try:\n",
    "    ee.Initialize()\n",
    "except Exception as e:\n",
    "    ee.Authenticate()\n",
    "    ee.Initialize()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Define Region of Interest and Timeframes\n",
    "We select an area in Sindh, Pakistan, highly affected by the summer 2022 floods."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the Area of Interest (AOI)\n",
    "aoi = ee.Geometry.Polygon(\n",
    "    [[[67.5, 25.5],\n",
    "      [67.5, 27.5],\n",
    "      [69.5, 27.5],\n",
    "      [69.5, 25.5]]]\n",
    ")\n",
    "\n",
    "# Define Timeframes\n",
    "pre_start = '2022-05-01'\n",
    "pre_end = '2022-06-30'\n",
    "\n",
    "post_start = '2022-08-15'\n",
    "post_end = '2022-09-15'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Retrieve and Filter Sentinel-1 SAR Imagery"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "collection = ee.ImageCollection('COPERNICUS/S1_GRD') \\\n",
    "    .filterBounds(aoi) \\\n",
    "    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \\\n",
    "    .filter(ee.Filter.eq('instrumentMode', 'IW'))\n",
    "\n",
    "pre_flood = collection.filterDate(pre_start, pre_end).mosaic().clip(aoi)\n",
    "post_flood = collection.filterDate(post_start, post_end).mosaic().clip(aoi)\n",
    "\n",
    "pre_vh = pre_flood.select('VH')\n",
    "post_vh = post_flood.select('VH')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Apply Speckle Filtering & Calculate Flood Extent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "smoothing_radius = 50 # meters\n",
    "pre_smoothed = pre_vh.focal_mean(smoothing_radius, 'circle', 'meters')\n",
    "post_smoothed = post_vh.focal_mean(smoothing_radius, 'circle', 'meters')\n",
    "\n",
    "# Difference map\n",
    "difference = post_smoothed.subtract(pre_smoothed)\n",
    "\n",
    "# Thresholds based on backscatter properties of water\n",
    "flood_threshold = -4\n",
    "water_threshold = -18\n",
    "\n",
    "flooded = difference.lt(flood_threshold).And(post_smoothed.lt(water_threshold))\n",
    "\n",
    "flood_mask = flooded.updateMask(flooded)\n",
    "\n",
    "# Calculate area\n",
    "area = flood_mask.multiply(ee.Image.pixelArea()).divide(10000) # Hectares\n",
    "try:\n",
    "    stats = area.reduceRegion(\n",
    "        reducer=ee.Reducer.sum(),\n",
    "        geometry=aoi,\n",
    "        scale=100,\n",
    "        maxPixels=1e10\n",
    "    )\n",
    "    print('Estimated Flooded Area (Hectares):', stats.getInfo()['VH'])\n",
    "except Exception as e:\n",
    "    print(\"Could not calculate exact area:\", e)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "Map = geemap.Map(center=[26.5, 68.5], zoom=8)\n",
    "\n",
    "Map.addLayer(pre_smoothed, {'min': -25, 'max': -5}, 'Pre-flood SAR', False)\n",
    "Map.addLayer(post_smoothed, {'min': -25, 'max': -5}, 'Post-flood SAR', False)\n",
    "Map.addLayer(flood_mask, {'palette': ['blue']}, 'Flood Extent')\n",
    "\n",
    "Map.addLayerControl()\n",
    "Map"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

project_dir = r"c:\Users\damil\OneDrive\Documents\Flood_Analysis_Project"
with open(os.path.join(project_dir, 'flood_analysis.ipynb'), 'w') as f:
    json.dump(notebook, f, indent=2)

print("Created notebook!")
