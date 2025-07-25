import streamlit as st
import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
import zipfile

# Set page config
st.set_page_config(page_title="School Routing App", layout="wide")

# Load Data Section
@st.cache_data
def load_data():
    # Load road network for Troy
    G = ox.graph_from_place("Troy, Michigan, USA", network_type="drive")

    # Directly load the extracted Troy city boundary shapefile
    places = gpd.read_file("troy_boundary/tl_2019_26_place.shp").to_crs(epsg=4326)
    troy = places[places["NAME"] == "Troy"]

    # Load student centroids
    gdf_students = gpd.read_file("students_near_athens.geojson").to_crs(epsg=4326)

    # Filter students to within Troy city boundary
    gdf_students = gdf_students[gdf_students.within(troy.unary_union)].reset_index(drop=True)

    return G, gdf_students, troy

G, gdf_students, troy = load_data()

# Get nearest graph node to Athens High School
school_lat, school_lon = 42.5841, -83.1250
depot_node = ox.distance.nearest_nodes(G, school_lon, school_lat)

# Sidebar Inputs
st.sidebar.header("Routing Parameters")

bus_capacities = st.sidebar.text_input("Bus Capacities (comma-separated)", value="40,40")
van_capacities = st.sidebar.text_input("Van Capacities (comma-separated)", value="9,9,9")
num_students = st.sidebar.slider("Number of Students", min_value=20, max_value=500, value=110, step=10)
bus_radius = st.sidebar.slider("Bus Radius (meters)", min_value=200, max_value=1000, value=550, step=50)
van_radius = st.sidebar.slider("Van Radius (meters)", min_value=50, max_value=500, value=150, step=25)

run_button = st.sidebar.button("Run Combined Routing")

# Display summary
st.title("School Bus & Van Routing Tool")
st.write(f"**Selected Students:** {num_students}")
st.write(f"**Bus Capacities:** {bus_capacities}")
st.write(f"**Van Capacities:** {van_capacities}")
st.write(f"**Bus Stop Radius:** {bus_radius} meters")
st.write(f"**Van Grouping Radius:** {van_radius} meters")

# Display school location on map
school_point = gpd.GeoDataFrame(geometry=[Point(school_lon, school_lat)], crs="EPSG:4326")
st.map(school_point)

if run_button:
    st.success("Routing will be initiated next...")
    # The next step is to call the routing functions you'll provide
