import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # For Streamlit compatibility (optional but safe)



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

    # --- VAN ASSIGNMENT ---
    # Identify leftover students
    remaining_students = gdf_students_sampled[~gdf_students_sampled.index.isin(bus_assigned_students)].reset_index(drop=True)
    st.markdown("### 🚐 Van Routing Summary")
    st.write(f"**Remaining Students for Vans:** {len(remaining_students)}")

    # DBSCAN for vans (small radius to allow occasional shared stops)
    van_radius_meters = van_radius
    coords_van = np.array(list(zip(remaining_students.geometry.y, remaining_students.geometry.x)))
    db_van = DBSCAN(eps=van_radius_meters / 111320, min_samples=2, metric='euclidean').fit(coords_van)

    remaining_students['van_cluster'] = db_van.labels_

    van_pickups = []
    van_assigned_count = 0

    # Original van fleet
    original_van_fleet = [int(x.strip()) for x in van_capacities.split(',')]
    van_capacity = max(original_van_fleet) if original_van_fleet else 10  # fallback capacity
    van_fleet_final = list(original_van_fleet)  # This will be expanded if needed

    # Build van stops
    for idx, row in remaining_students.iterrows():
        if row['van_cluster'] == -1:
            # Door-to-door pickup (isolated student)
            geom = row.geometry
            nearest_node = ox.distance.nearest_nodes(G, geom.x, geom.y)
            van_pickups.append({
                'students': [idx],
                'geometry': geom,
                'demand': 1
            })
            van_assigned_count += 1
        else:
            # Small shared stop
            group = remaining_students[remaining_students['van_cluster'] == row['van_cluster']]
            if group.index[0] != idx:
                continue  # Prevent duplicate stop creation for the same cluster

            centroid = Point(group.geometry.x.mean(), group.geometry.y.mean())
            nearest_node = ox.distance.nearest_nodes(G, centroid.x, centroid.y)
            x = G.nodes[nearest_node]['x']
            y = G.nodes[nearest_node]['y']
            geom = Point(x, y)
            demand = len(group.index)

            for student_idx in group.index:
                van_pickups.append({
                    'students': [student_idx],
                    'geometry': geom,
                    'demand': 1  # Keep demand per student for vans (simplify routing)
                })
                van_assigned_count += 1

    # Auto-expand van fleet if necessary
    total_van_demand = sum([stop['demand'] for stop in van_pickups])
    while total_van_demand > sum(van_fleet_final):
        van_fleet_final.append(van_capacity)

    st.write(f"**Total Van Stops Created:** {len(van_pickups)}")
    st.write(f"**Total Students Assigned to Vans:** {total_van_demand}")
    st.write(f"**Van Fleet After Expansion:** {van_fleet_final}")

    # Save stops for vans
    gdf_van_stops = gpd.GeoDataFrame(van_pickups, crs=remaining_students.crs)
    gdf_van_stops['osmid'] = gdf_van_stops.geometry.apply(lambda pt: ox.distance.nearest_nodes(G, pt.x, pt.y))

    # --- OR-Tools VRP Solver ---
    def solve_vrp(G, stops_df, depot_node, fleet_capacities, fleet_type='Vehicle'):
        stop_nodes = list(stops_df["osmid"])
        all_nodes = [depot_node] + stop_nodes

        num_locations = len(all_nodes)
        distance_matrix = np.zeros((num_locations, num_locations))
        for i in range(num_locations):
            for j in range(num_locations):
                if i == j:
                    distance_matrix[i][j] = 0
                else:
                    try:
                        length = nx.shortest_path_length(G, all_nodes[i], all_nodes[j], weight='length')
                        distance_matrix[i][j] = length
                    except:
                        distance_matrix[i][j] = 1e6  # unreachable penalty

        demands = [0] + list(stops_df["demand"])

        manager = pywrapcp.RoutingIndexManager(num_locations, len(fleet_capacities), 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, fleet_capacities, True, 'Capacity'
        )

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 60

        solution = routing.SolveWithParameters(search_parameters)

        vehicle_routes = {}
        if solution:
            for vehicle_id in range(len(fleet_capacities)):
                index = routing.Start(vehicle_id)
                route = []
                while not routing.IsEnd(index):
                    route.append(manager.IndexToNode(index))
                    index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))  # end at depot
                vehicle_routes[vehicle_id] = {"route": route}

            st.success(f"✅ **{fleet_type} VRP Solved:** {len(vehicle_routes)} {fleet_type.lower()}(s) routed")
        else:
            st.error(f"⚠️ {fleet_type} VRP could not be solved within time limit.")

        return vehicle_routes, all_nodes


    # Solve separately
    st.markdown("### 🛣️ Route Optimization with OR-Tools")

    bus_routes, bus_nodes = solve_vrp(G, gdf_bus_stops, depot_node, bus_fleet, fleet_type='Bus')
    van_routes, van_nodes = solve_vrp(G, gdf_van_stops, depot_node, van_fleet_final, fleet_type='Van')



    # --- VEHICLE ASSIGNMENT SUMMARY TABLES ---
    def get_vehicle_assignment_df(vehicle_routes, stops_df, fleet_type="Vehicle", start_id=1):
        data = []
        for vehicle_id, route_info in vehicle_routes.items():
            stop_indices = [idx for idx in route_info["route"] if idx != 0]
            student_count = 0
            for idx in stop_indices:
                stop_df_idx = idx - 1
                demand_at_stop = stops_df.iloc[stop_df_idx]['demand']
                student_count += demand_at_stop

            data.append({
                "Vehicle": f"{fleet_type} {vehicle_id + start_id}",
                "Students Served": student_count
            })
        return pd.DataFrame(data)

    # Generate assignment tables
    bus_assignment_df = get_vehicle_assignment_df(bus_routes, gdf_bus_stops, fleet_type="Bus", start_id=1)
    van_assignment_df = get_vehicle_assignment_df(van_routes, gdf_van_stops, fleet_type="Van", start_id=1)
    assignment_df = pd.concat([bus_assignment_df, van_assignment_df]).reset_index(drop=True)

    # Create Plotly table
    fig_assignment = ff.create_table(
        assignment_df,
        colorscale=[[0, '#cc0000'], [1, '#ffdddd']]  # dark red to light red
    )
    fig_assignment.update_layout(title_text="🚍 Vehicle Assignment Summary", title_x=0.5)

    # Display in Streamlit
    st.markdown("### 📋 Vehicle Assignments Overview")
    st.plotly_chart(fig_assignment, use_container_width=True)

    # --- BUS STOP CLUSTERING EVALUATION PLOT ---
    st.markdown("### 🧭 Bus Pick-Up Stops vs Student Locations")

    # Generate the base graph
    fig, ax = ox.plot_graph(G, show=False, close=False, bgcolor='black', node_size=0)

    # Plot all students as small white dots
    ax.scatter(gdf_students_sampled.geometry.x, gdf_students_sampled.geometry.y,
               color='white', s=3, label='Students')

    # Generate color palette for buses
    colors = plt.cm.tab10.colors if len(bus_routes) <= 10 else plt.cm.nipy_spectral(np.linspace(0, 1, len(bus_routes)))

    # Plot bus stops and routes by color
    for bus_id, route_info in bus_routes.items():
        stop_indices = [idx for idx in route_info["route"] if idx != 0]
        color = colors[bus_id % len(colors)]

        for stop_idx in stop_indices:
            stop_df_idx = stop_idx - 1
            stop_geom = gdf_bus_stops.iloc[stop_df_idx].geometry
            ax.scatter(stop_geom.x, stop_geom.y,
                       color=color, s=50, label=f'Bus {bus_id} Stops')

    # Clean up duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', fontsize='small', frameon=False)

    # Add title and push to Streamlit
    plt.title('Shared Bus Stops Created From Student Clusters', color='white')
    st.pyplot(fig)


