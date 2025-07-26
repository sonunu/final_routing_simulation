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
import io
import matplotlib.animation as animation
# from IPython.display import HTML  # still used to get HTML5 video string
import streamlit.components.v1 as components


# Set page config
st.set_page_config(page_title="School Routing App", layout="wide")

# Load Data Section
@st.cache_data
def load_data():
    # Load road network for Troy
    G = ox.graph_from_place("Troy, Michigan, USA", network_type="drive")
    gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)

    # Directly load the extracted Troy city boundary shapefile
    places = gpd.read_file("tl_2019_26_place.shp").to_crs(epsg=4326)
    troy = places[places["NAME"] == "Troy"]

    # Load student centroids
    gdf_students = gpd.read_file("students_near_athens.geojson").to_crs(epsg=4326)

    # Filter students to within Troy city boundary
    gdf_students = gdf_students[gdf_students.within(troy.union_all())].reset_index(drop=True)

    return G, gdf_students, troy, gdf_nodes

G, gdf_students, troy, gdf_nodes = load_data()

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
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', fontsize='small', frameon=True, facecolor='white', labelcolor='black')

    # Add title and push to Streamlit
    plt.title('Shared Bus Stops Created From Student Clusters', color='white')
    st.pyplot(fig)

    # --- HELPER FUNCTION: STOP DETAIL TABLES ---
    def get_stop_details_df(vehicle_routes, all_nodes, stops_df, vehicle_type="Vehicle", start_id=1):
        rows = []
        for vehicle_id, route_info in vehicle_routes.items():
            route = route_info["route"]
            stop_indices = [idx for idx in route if idx != 0]
            for stop_order, stop_idx in enumerate(stop_indices, start=1):
                stop_df_idx = stop_idx - 1
                stop_osmid = all_nodes[stop_idx]
                demand = stops_df.iloc[stop_df_idx]['demand']
                rows.append({
                    "Vehicle": f"{vehicle_type} {vehicle_id + start_id}",
                    "Stop Order": stop_order,
                    "Stop Node": stop_osmid,
                    "Students at Stop": demand
                })
        return pd.DataFrame(rows)

    # --- CREATE BUS + VAN STOP DETAIL TABLES ---
    bus_stop_details_df = get_stop_details_df(bus_routes, bus_nodes, gdf_bus_stops, vehicle_type="Bus", start_id=1)
    van_stop_details_df = get_stop_details_df(van_routes, van_nodes, gdf_van_stops, vehicle_type="Van", start_id=1)

    # --- PLOTLY TABLE: BUS STOP DETAILS ---
    fig_bus_stops = ff.create_table(
        bus_stop_details_df,
        colorscale=[[0, '#cc0000'], [1, '#ffdddd']]
    )
    fig_bus_stops.update_layout(title_text="🚌 Bus Stop Details", title_x=0.5)
    st.markdown("### 🚌 Bus Stop Details")
    st.plotly_chart(fig_bus_stops, use_container_width=True)

    # --- PLOTLY TABLE: VAN STOP DETAILS ---
    fig_van_stops = ff.create_table(
        van_stop_details_df,
        colorscale=[[0, '#cc0000'], [1, '#ffdddd']]
    )
    fig_van_stops.update_layout(title_text="🚐 Van Stop Details", title_x=0.5)
    st.markdown("### 🚐 Van Stop Details")
    st.plotly_chart(fig_van_stops, use_container_width=True)

    # --- BUS + VAN STOPS PER VEHICLE MAP ---
    st.markdown("### 🗺️ Bus and Van Stops by Vehicle")

    # Prepare the figure
    fig, ax = ox.plot_graph(G, show=False, close=False, bgcolor='black', node_size=0)

    # Colormap for vehicles
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(bus_routes) + len(van_routes))]

    # Plot bus stops
    for bus_id, route_info in bus_routes.items():
        stop_indices = [idx for idx in route_info["route"] if idx != 0]
        color = colors[bus_id]

        for stop_idx in stop_indices:
            stop_df_idx = stop_idx - 1
            stop_geom = gdf_bus_stops.iloc[stop_df_idx].geometry
            ax.scatter(stop_geom.x, stop_geom.y,
                       color=color, s=50, label=f'Bus {bus_id + 1}')

    # Plot van stops
    offset = len(bus_routes)
    for van_id, route_info in van_routes.items():
        stop_indices = [idx for idx in route_info["route"] if idx != 0]
        color = colors[offset + van_id]

        for stop_idx in stop_indices:
            stop_df_idx = stop_idx - 1
            stop_geom = gdf_van_stops.iloc[stop_df_idx].geometry
            ax.scatter(stop_geom.x, stop_geom.y,
                       color=color, s=50, label=f'Van {van_id + 1}')

    # Handle duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', fontsize='small', frameon=True, facecolor='white', labelcolor='black')

    plt.title('Bus + Van Stops per Vehicle', color='white')
    st.pyplot(fig)

    # --- DISTANCE & TIME SUMMARY TABLE ---

    def compute_distance_time(vehicle_routes, all_nodes, G, vehicle_type, start_id=1, average_speed_kmph=30):
        summary_rows = []
        meters_per_km = 1000

        for vehicle_id, route_info in vehicle_routes.items():
            route = route_info["route"]
            route_osmids = [all_nodes[idx] for idx in route]

            total_distance_m = 0

            for u, v in zip(route_osmids[:-1], route_osmids[1:]):
                try:
                    length = nx.shortest_path_length(G, u, v, weight='length')
                    total_distance_m += length
                except:
                    continue  # Skip disconnected pairs

            total_distance_km = total_distance_m / meters_per_km
            total_time_minutes = (total_distance_km / average_speed_kmph) * 60

            summary_rows.append({
                'Vehicle': f"{vehicle_type} {vehicle_id + start_id}",
                'Type': vehicle_type,
                'Distance (km)': round(total_distance_km, 2),
                'Time (min)': round(total_time_minutes, 1)
            })

        return summary_rows

    # Compute summary for buses and vans
    bus_summary = compute_distance_time(bus_routes, bus_nodes, G, vehicle_type="Bus")
    van_summary = compute_distance_time(van_routes, van_nodes, G, vehicle_type="Van", start_id=1)

    # Combine and sort
    summary_df = pd.DataFrame(bus_summary + van_summary)
    summary_df_sorted = summary_df.sort_values(by="Vehicle").reset_index(drop=True)

    # Create Plotly table
    fig_summary = ff.create_table(
        summary_df_sorted,
        colorscale=[[0, '#cc0000'], [1, '#ffdddd']]  # red theme
    )

    fig_summary.update_layout(title_text="📏 Vehicle Route Distance & Time Summary", title_x=0.5)

    # Display in Streamlit
    st.markdown("### ⏱️ Distance and Time for Each Route")
    st.plotly_chart(fig_summary, use_container_width=True)

    # --- GENERATE PER-STOP ROUTE DATA ---
    def generate_per_stop_data(vehicle_routes, all_nodes, stops_df, gdf_nodes, vehicle_type, start_id=1):
        per_stop_rows = []

        for vehicle_id, route_info in vehicle_routes.items():
            route = route_info["route"]
            stop_indices = [idx for idx in route if idx != 0]  # skip depot

            stop_order = 1
            for stop_idx in stop_indices:
                stop_df_idx = stop_idx - 1  # depot is index 0
                stop_osmid = all_nodes[stop_idx]
                demand = stops_df.iloc[stop_df_idx]['demand']

                # Get coordinates
                point_geom = gdf_nodes.loc[stop_osmid].geometry
                lat = point_geom.y
                lon = point_geom.x

                per_stop_rows.append({
                    'Vehicle': f"{vehicle_type} {vehicle_id + start_id}",
                    'Type': vehicle_type,
                    'Stop Order': stop_order,
                    'Stop Node': stop_osmid,
                    'Students Picked Up': demand,
                    'Latitude': lat,
                    'Longitude': lon
                })

                stop_order += 1

        return per_stop_rows

    # Build for buses and vans
    bus_per_stop = generate_per_stop_data(bus_routes, bus_nodes, gdf_bus_stops, gdf_nodes, vehicle_type="Bus")
    van_per_stop = generate_per_stop_data(van_routes, van_nodes, gdf_van_stops, gdf_nodes, vehicle_type="Van")
    per_stop_df = pd.DataFrame(bus_per_stop + van_per_stop)

    # --- EXCEL DOWNLOAD BUTTON ---
    st.markdown("### 📥 Download Route Data")

    # Convert DataFrame to Excel file in memory
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        per_stop_df.to_excel(writer, index=False, sheet_name='Route Details')
        
        excel_data = excel_buffer.getvalue()

    # Download button
    st.download_button(
        label="Download Excel File with Route Details",
        data=excel_data,
        file_name="per_stop_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


    # --- PLOTLY MAP: PER VEHICLE STOP LOCATIONS WITH DROPDOWN ---

    st.markdown("### 🗺️ Interactive Map of Bus & Van Stops")

    # Sort by stop order
    per_stop_df_sorted = per_stop_df.sort_values(by=["Vehicle", "Stop Order"])

    # Create figure
    fig = go.Figure()
    visibility_all = []
    visibility_buses = []
    visibility_vans = []

    # Plot each vehicle’s stop sequence
    for vehicle in per_stop_df_sorted['Vehicle'].unique():
        vehicle_df = per_stop_df_sorted[per_stop_df_sorted['Vehicle'] == vehicle]
        vehicle_type = vehicle_df['Type'].iloc[0]

        fig.add_trace(go.Scattermapbox(
            lat=vehicle_df["Latitude"],
            lon=vehicle_df["Longitude"],
            mode="markers+text",
            marker=dict(size=10),
            text=vehicle_df["Stop Order"].astype(str),
            name=vehicle,
            legendgroup=vehicle_type,
            showlegend=True,
            textposition="top center",
            hoverinfo="text",
            hovertext=[
                f"{vehicle}<br>"
                f"Stop Number: {row['Stop Order']}<br>"
                f"Latitude: {row['Latitude']:.5f}<br>"
                f"Longitude: {row['Longitude']:.5f}<br>"
                f"Students Picked Up: {row['Students Picked Up']}"
                for i, row in vehicle_df.iterrows()
            ]
        ))

        visibility_all.append(True)
        visibility_buses.append(vehicle_type == "Bus")
        visibility_vans.append(vehicle_type == "Van")

    # Add school as red star
    fig.add_trace(go.Scattermapbox(
        lat=[school_lat],
        lon=[school_lon],
        mode="markers",
        marker=dict(size=30, symbol="star", color="red"),
        name="School",
        hoverinfo="text",
        hovertext=["School"],
        showlegend=True
    ))

    # Ensure school is visible in all views
    visibility_all.append(True)
    visibility_buses.append(True)
    visibility_vans.append(True)

    # Map style/layout
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=12,
        mapbox_center={"lat": per_stop_df["Latitude"].mean(), "lon": per_stop_df["Longitude"].mean()},
        height=800,
        title="🧭 Bus and Van Pickup Stops by Vehicle",
    )

    # Add dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                x=0.01,
                y=1.05,
                buttons=[
                    dict(label="Show All", method="update", args=[{"visible": visibility_all}]),
                    dict(label="Only Buses", method="update", args=[{"visible": visibility_buses}]),
                    dict(label="Only Vans", method="update", args=[{"visible": visibility_vans}]),
                ],
                direction="down",
                showactive=True,
            )
        ]
    )

    # Display map
    st.plotly_chart(fig, use_container_width=True)

    # === Route Animation Section ===
    st.markdown("## 📽️ Route Animations")

    col1, col2 = st.columns(2)

    with col1:
        prototype_clicked = st.button("🎞️ Generate Prototype Animation")
        st.caption("Use if you want a quick demo of how the routes will move along their paths! (shorter loading time but less accurate routes)")

    with col2:
        google_maps_clicked = st.button("🗺️ Generate Google Maps Animation")
        st.caption("Use if you want a more professional route display of how the routes will operate! (longer loading time + Google Directions and Google Maps API needed)")

    # Run animations based on user click
    if prototype_clicked:
        st.markdown("### 🎞️ Prototype Route Animation")
        st.info("Generating animation... Please wait ⏳")

        # Base map
        fig, ax = ox.plot_graph(G, show=False, close=False, bgcolor='white', node_size=0)
        cmap = plt.colormaps.get_cmap('tab10')
        num_vehicles = len(bus_routes) + len(van_routes)
        colors = [cmap(i % 10) for i in range(num_vehicles)]

        all_vehicle_paths = []
        vehicle_labels = []

        # Plot school
        x, y = gdf_nodes.loc[depot_node].geometry.xy
        ax.plot(x[0], y[0], marker='*', color='red', markersize=20, label='Athens High School')

        # --- BUS ROUTES ---
        for vid, route_info in bus_routes.items():
            color = colors[vid]
            label = f"Bus {vid+1}"
            vehicle_labels.append(label)

            route_osmids = [bus_nodes[idx] for idx in route_info["route"]]
            full_path = []

            for idx in route_info["route"]:
                if idx == 0:
                    continue
                osmid = bus_nodes[idx]
                point_geom = gdf_nodes.loc[osmid].geometry
                ax.plot(point_geom.x, point_geom.y, marker='o', color=color, markersize=5, alpha=0.9)

            for u, v in zip(route_osmids[:-1], route_osmids[1:]):
                try:
                    segment = nx.shortest_path(G, u, v, weight='length')
                    full_path.extend(segment[:-1])
                except:
                    continue
            full_path.append(route_osmids[-1])
            all_vehicle_paths.append(full_path)

        # --- VAN ROUTES ---
        offset = len(bus_routes)
        for vid, route_info in van_routes.items():
            color = colors[offset + vid]
            label = f"Van {vid+1}"
            vehicle_labels.append(label)

            route_osmids = [van_nodes[idx] for idx in route_info["route"]]
            full_path = []

            for idx in route_info["route"]:
                if idx == 0:
                    continue
                osmid = van_nodes[idx]
                point_geom = gdf_nodes.loc[osmid].geometry
                ax.plot(point_geom.x, point_geom.y, marker='o', color=color, markersize=5, alpha=0.9)

            for u, v in zip(route_osmids[:-1], route_osmids[1:]):
                try:
                    segment = nx.shortest_path(G, u, v, weight='length')
                    full_path.extend(segment[:-1])
                except:
                    continue
            full_path.append(route_osmids[-1])
            all_vehicle_paths.append(full_path)

        # Legend
        for idx, label in enumerate(vehicle_labels):
            ax.plot([], [], color=colors[idx], label=label)
        ax.legend(loc='upper left', fontsize='small')

        # --- ANIMATION SETUP ---
        dots, lines, trails = [], [], []
        for idx, path in enumerate(all_vehicle_paths):
            color = colors[idx]
            dot, = ax.plot([], [], marker='o', color=color, markersize=8)
            line, = ax.plot([], [], color=color, linewidth=2)
            dots.append(dot)
            lines.append(line)
            trails.append(([], []))

        def update(frame):
            for idx, path in enumerate(all_vehicle_paths):
                if frame < len(path):
                    node_id = path[frame]
                else:
                    node_id = path[-1]
                point_geom = gdf_nodes.loc[node_id].geometry
                x, y = point_geom.x, point_geom.y
                dots[idx].set_data([x], [y])
                trails[idx][0].append(x)
                trails[idx][1].append(y)
                lines[idx].set_data(trails[idx][0], trails[idx][1])
            return dots + lines

        max_frames = max(len(path) for path in all_vehicle_paths)
        ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=100, blit=False, repeat=False)

        # Save the animation as an MP4 video
        ani.save("prototype_animation.mp4", writer="ffmpeg")

        # Display the MP4 video using Streamlit's built-in player
        with open("prototype_animation.mp4", "rb") as f:
            st.video(f.read())

