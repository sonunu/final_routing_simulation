import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
import os
import json
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from io import BytesIO

# --- Set Config ---
st.set_page_config(page_title="School Bus & Van Routing App", layout="wide")
st.title("🚌 Combined Bus & Van Routing for Athens High School")

# --- Sidebar Inputs ---
bus_cap_input = st.sidebar.text_input("Bus Capacities (comma-separated)", "50,50")
van_cap_input = st.sidebar.text_input("Van Capacities (comma-separated)", "10,10,10")
num_students = st.sidebar.slider("Number of Students", 20, 500, 200, step=10)
bus_radius = st.sidebar.slider("Bus Radius (meters)", 200, 1000, 400, step=50)
run_button = st.sidebar.button("🚦 Run Routing")

# --- Helper Functions ---
@st.cache_data
def load_data():
    G = ox.graph_from_place("Troy, Michigan, USA", network_type="drive")
    gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    gdf_students = gpd.read_file("students_near_athens.geojson").to_crs(epsg=4326)
    boundary = gpd.read_file("tl_2019_26_place.shp")
    boundary = boundary[boundary.NAME == "Troy"].to_crs(epsg=4326)
    gdf_students = gdf_students[gdf_students.within(boundary.unary_union)]
    return G, gdf_nodes, gdf_students

def solve_vrp(G, stops_df, depot_node, fleet_capacities):
    stop_nodes = list(stops_df["osmid"])
    all_nodes = [depot_node] + stop_nodes
    demands = [0] + list(stops_df["demand"])

    matrix = np.zeros((len(all_nodes), len(all_nodes)))
    for i in range(len(all_nodes)):
        for j in range(len(all_nodes)):
            if i != j:
                try:
                    matrix[i][j] = nx.shortest_path_length(G, all_nodes[i], all_nodes[j], weight="length")
                except:
                    matrix[i][j] = 1e6

    manager = pywrapcp.RoutingIndexManager(len(all_nodes), len(fleet_capacities), 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_idx, to_idx):
        return int(matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)])

    def demand_cb(from_idx):
        return demands[manager.IndexToNode(from_idx)]

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_cb))
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_cb), 0, fleet_capacities, True, 'Capacity')

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 60

    solution = routing.SolveWithParameters(params)
    routes = {}
    if solution:
        for v in range(len(fleet_capacities)):
            index = routing.Start(v)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            routes[v] = {"route": route}
    return routes, all_nodes

# --- Main Logic ---
if run_button:
    with st.spinner("Running routing and generating files..."):
        G, gdf_nodes, gdf_students = load_data()
        depot_node = ox.distance.nearest_nodes(G, -83.1250, 42.5841)

        gdf_students = gdf_students.sample(n=num_students, random_state=42).reset_index(drop=True)
        coords_bus = np.array(list(zip(gdf_students.geometry.y, gdf_students.geometry.x)))
        db = DBSCAN(eps=bus_radius / 111320, min_samples=2).fit(coords_bus)
        gdf_students['bus_cluster'] = db.labels_

        bus_cap = [int(x.strip()) for x in bus_cap_input.split(',') if x.strip().isdigit()]
        van_cap = [int(x.strip()) for x in van_cap_input.split(',') if x.strip().isdigit()]

        total_bus = sum(bus_cap)
        assigned = 0
        bus_pickups, assigned_students = [], []

        for label in sorted(gdf_students['bus_cluster'].unique()):
            if label == -1: continue
            group = gdf_students[gdf_students['bus_cluster'] == label]
            if assigned + len(group) > total_bus: continue
            assigned_students.extend(group.index)
            assigned += len(group)
            centroid = Point(group.geometry.x.mean(), group.geometry.y.mean())
            node = ox.distance.nearest_nodes(G, centroid.x, centroid.y)
            geom = Point(G.nodes[node]['x'], G.nodes[node]['y'])
            bus_pickups.append({'students': list(group.index), 'geometry': geom, 'demand': len(group)})

        gdf_bus = gpd.GeoDataFrame(bus_pickups, crs=gdf_students.crs)
        gdf_bus['osmid'] = gdf_bus.geometry.apply(lambda pt: ox.distance.nearest_nodes(G, pt.x, pt.y))

        rem_students = gdf_students[~gdf_students.index.isin(assigned_students)]
        coords_van = np.array(list(zip(rem_students.geometry.y, rem_students.geometry.x)))
        db_van = DBSCAN(eps=150 / 111320, min_samples=2).fit(coords_van)
        rem_students['van_cluster'] = db_van.labels_

        van_pickups = []
        for idx, row in rem_students.iterrows():
            if row['van_cluster'] == -1:
                geom = row.geometry
                van_pickups.append({'students': [idx], 'geometry': geom, 'demand': 1})
            else:
                group = rem_students[rem_students['van_cluster'] == row['van_cluster']]
                if group.index[0] != idx: continue
                centroid = Point(group.geometry.x.mean(), group.geometry.y.mean())
                node = ox.distance.nearest_nodes(G, centroid.x, centroid.y)
                geom = Point(G.nodes[node]['x'], G.nodes[node]['y'])
                for sid in group.index:
                    van_pickups.append({'students': [sid], 'geometry': geom, 'demand': 1})

        gdf_van = gpd.GeoDataFrame(van_pickups, crs=rem_students.crs)
        gdf_van['osmid'] = gdf_van.geometry.apply(lambda pt: ox.distance.nearest_nodes(G, pt.x, pt.y))

        while sum(van_cap) < len(van_pickups):
            van_cap.append(max(van_cap))

        bus_routes, bus_nodes = solve_vrp(G, gdf_bus, depot_node, bus_cap)
        van_routes, van_nodes = solve_vrp(G, gdf_van, depot_node, van_cap)

        # Build JSON outputs
        route_json, student_json = [], []

        for b, route in bus_routes.items():
            coords = [(gdf_nodes.loc[bus_nodes[i]].geometry.y, gdf_nodes.loc[bus_nodes[i]].geometry.x) for i in route['route']]
            route_json.append({"vehicle": f"Bus {b+1}", "coordinates": coords})
        for v, route in van_routes.items():
            coords = [(gdf_nodes.loc[van_nodes[i]].geometry.y, gdf_nodes.loc[van_nodes[i]].geometry.x) for i in route['route']]
            route_json.append({"vehicle": f"Van {v+1}", "coordinates": coords})

        for b, route in bus_routes.items():
            for idx in route['route']:
                if idx == 0: continue
                stop = gdf_bus.iloc[idx-1]
                for sid in stop['students']:
                    pt = gdf_students.iloc[sid].geometry
                    student_json.append({"lat": pt.y, "lon": pt.x, "vehicle": f"Bus {b+1}"})
        for v, route in van_routes.items():
            for idx in route['route']:
                if idx == 0: continue
                stop = gdf_van.iloc[idx-1]
                for sid in stop['students']:
                    pt = gdf_students.iloc[sid].geometry
                    student_json.append({"lat": pt.y, "lon": pt.x, "vehicle": f"Van {v+1}"})

        with open("routes_for_google_maps.json", "w") as f:
            json.dump(route_json, f)
        with open("student_points_colored.json", "w") as f:
            json.dump(student_json, f)

        df_stops = pd.DataFrame([{
            "vehicle": r["vehicle"],
            "lat": pt[0],
            "lon": pt[1]
        } for r in route_json for pt in r["coordinates"]])
        df_students = pd.DataFrame(student_json)
        df_students.insert(0, "Simulated Student", [f"Student {i+1}" for i in range(len(df_students))])

        excel_bytes = BytesIO()
        with pd.ExcelWriter(excel_bytes, engine="xlsxwriter") as writer:
            df_stops.to_excel(writer, sheet_name="Stop Points", index=False)
            df_students.to_excel(writer, sheet_name="Student Assignments", index=False)

        st.success("✅ Routing complete!")

        st.header("🗺️ Route Maps")
        tab1, tab2 = st.tabs(["Animated Map", "Static Map"])
        with tab1:
            st.components.v1.html(open("animated_routes_with_stops.html").read(), height=700)
        with tab2:
            st.components.v1.html(open("routes_visualization.html").read(), height=700)

        st.download_button("📥 Download Route Report", data=excel_bytes.getvalue(), file_name="route_report.xlsx")
