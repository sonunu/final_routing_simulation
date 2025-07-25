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
    places = gpd.read_file("tl_2019_26_place.shp").to_crs(epsg=4326)
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
school_df = pd.DataFrame({
    'latitude': [school_lat],
    'longitude': [school_lon]
})
st.map(school_df)

if run_button:
    st.success("Routing will be initiated next...")

    # Parse fleet and radius
    bus_fleet = [int(x.strip()) for x in bus_capacities.split(',')]
    van_fleet = [int(x.strip()) for x in van_capacities.split(',')]
    bus_radius_meters = bus_radius

    # Total bus capacity
    total_bus_capacity = sum(bus_fleet)
    bus_assigned_count = 0

    # Sample students from dataset
    gdf_students_sampled = gdf_students.sample(n=num_students, random_state=42).reset_index(drop=True)

    # --- BUS CLUSTERING FIRST ---
    coords_bus = np.array(list(zip(gdf_students_sampled.geometry.y, gdf_students_sampled.geometry.x)))
    db_bus = DBSCAN(eps=bus_radius_meters / 111320, min_samples=2, metric='euclidean').fit(coords_bus)

    gdf_students_sampled['bus_cluster'] = db_bus.labels_

    bus_pickups = []
    bus_assigned_students = []

    for label in sorted(gdf_students_sampled['bus_cluster'].unique()):
        if label == -1:
            continue  # Skip unclustered students

        group = gdf_students_sampled[gdf_students_sampled['bus_cluster'] == label]
        group_size = len(group.index)

        if bus_assigned_count + group_size > total_bus_capacity:
            continue  # Skip this cluster to avoid overloading bus fleet

        # Assign cluster to buses
        bus_assigned_students.extend(group.index)
        bus_assigned_count += group_size

        # Create shared stop snapped to nearest intersection
        centroid = Point(group.geometry.x.mean(), group.geometry.y.mean())
        nearest_node = ox.distance.nearest_nodes(G, centroid.x, centroid.y)
        x = G.nodes[nearest_node]['x']
        y = G.nodes[nearest_node]['y']
        geom = Point(x, y)
        demand = group_size

        bus_pickups.append({
            'students': list(group.index),
            'geometry': geom,
            'demand': demand
        })

    # Save shared stops for buses
    gdf_bus_stops = gpd.GeoDataFrame(bus_pickups, crs=gdf_students_sampled.crs)
    gdf_bus_stops['osmid'] = gdf_bus_stops.geometry.apply(lambda pt: ox.distance.nearest_nodes(G, pt.x, pt.y))

    # Display summary stats
    st.markdown("### 🚌 Bus Routing Summary")
    st.write(f"**Total Bus Stops Created:** {len(gdf_bus_stops)}")
    st.write(f"**Total Students Assigned to Buses:** {bus_assigned_count} / {total_bus_capacity}")

