import streamlit as st
import sys
import os
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import random

# --- Path settings ---
# This ensures that the script can find the 'datagen' module
# Assumes 'datagen' is in the parent directory of the 'streamlit' directory
current_dir = os.path.dirname(__file__) # pages directory
streamlit_dir = os.path.abspath(os.path.join(current_dir, '..')) # streamlit directory
project_root_dir = os.path.abspath(os.path.join(streamlit_dir, '..')) # hercules directory (project root)
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

from datagen.data_utils import generate_graph # Import from the datagen module

st.set_page_config(layout="wide", page_title="Graph Visualization & Edge Probabilities")
st.title("Graph Visualization & Edge Propagation Probability Setting")

# --- Sidebar for graph generation ---
st.sidebar.header("Graph Generation Parameters")
# Using unique keys for widgets in this version to avoid conflicts if other versions exist
num_nodes = st.sidebar.slider("Number of Nodes", 5, 100, 20, key="gv_num_nodes_static")
edge_density = st.sidebar.slider("Edge Density", 0.01, 0.5, 0.1, key="gv_edge_density_static")
is_directed = st.sidebar.checkbox("Directed Graph", True, key="gv_is_directed_static")

# Graph generation/regeneration logic
# Using unique session state keys for this static version
graph_key_base_static = 'gv_graph_base_static'
graph_key_weighted_static = 'gv_graph_with_weights_static'

# Generate graph if parameters change or base graph doesn't exist in session state
if graph_key_base_static not in st.session_state or \
   st.session_state.get('gv_num_nodes_param_static') != num_nodes or \
   st.session_state.get('gv_edge_density_param_static') != edge_density or \
   st.session_state.get('gv_is_directed_param_static') != is_directed:

    st.session_state['gv_num_nodes_param_static'] = num_nodes
    st.session_state['gv_edge_density_param_static'] = edge_density
    st.session_state['gv_is_directed_param_static'] = is_directed
    
    G_base = generate_graph(num_nodes, edge_density, directed=is_directed)
    st.session_state[graph_key_base_static] = G_base
    
    # If base graph changes, reset the weighted graph from session state if it exists
    if graph_key_weighted_static in st.session_state:
        del st.session_state[graph_key_weighted_static]
    st.sidebar.info("Graph parameters changed. Graph regenerated. Please re-assign propagation probabilities if needed.")
else:
    # Use existing graph from session state
    G_base = st.session_state[graph_key_base_static]

# --- Sidebar for edge propagation probability settings ---
st.sidebar.markdown("---")
st.sidebar.header("Edge Propagation Probability Settings")
prob_min = st.sidebar.slider("Min Propagation Probability", 0.0, 1.0, 0.01, 0.01, key="gv_prob_min_static")
prob_max = st.sidebar.slider("Max Propagation Probability", 0.0, 1.0, 0.2, 0.01, key="gv_prob_max_static")

assign_disabled = False
if prob_min > prob_max:
    st.sidebar.error("Min propagation probability must be less than or equal to max propagation probability.")
    assign_disabled = True

if st.sidebar.button("Assign Random Propagation Probabilities to Edges", disabled=assign_disabled, key="gv_assign_prob_button_static"):
    if G_base and G_base.number_of_nodes() > 0:
        G_weighted = G_base.copy() # Work on a copy of the base graph
        for u, v in G_weighted.edges():
            # Assign a random weight (probability) to each edge within the specified range
            G_weighted[u][v]['weight'] = random.uniform(prob_min, prob_max)
        st.session_state[graph_key_weighted_static] = G_weighted
        st.sidebar.success(f"Propagation probabilities (randomly between {prob_min:.2f} and {prob_max:.2f}) assigned to all edges.")
    elif not G_base: # Check if G_base is None
        st.sidebar.warning("Please generate a graph first using the parameters above.")
    else: # G_base exists but has no nodes
        st.sidebar.warning("The generated graph has no nodes. Please adjust generation parameters.")


# --- Main page display ---
# Select graph to display: weighted if available, otherwise the base graph
G_to_display = st.session_state.get(graph_key_weighted_static, G_base)

if G_to_display:
    st.write(f"Displaying graph: {G_to_display.number_of_nodes()} nodes, {G_to_display.number_of_edges()} edges.")
    if graph_key_weighted_static in st.session_state and st.session_state[graph_key_weighted_static] is G_to_display:
        # Check if the displayed graph is the one with weights
        # Get the min/max probabilities that were used when weights were assigned (if stored)
        # For simplicity, we'll just state they are assigned.
        # To show the actual range used, you'd need to store prob_min and prob_max at the time of assignment.
        st.success(f"Edge propagation probabilities have been assigned.")
    else:
        st.info("Edge propagation probabilities have not been assigned yet. Use the sidebar to assign them.")

    # Prepare nodes and edges for streamlit-agraph
    nodes_vis = []
    for node_id in G_to_display.nodes():
        nodes_vis.append(Node(id=str(node_id), label=str(node_id), size=15)) # Basic node visualization

    edges_vis = []
    for u, v, data in G_to_display.edges(data=True): # Get edge data to access 'weight'
        edge_label = ""
        if 'weight' in data:
            edge_label = f"{data['weight']:.2f}" # Display weight on edge if it exists
        edges_vis.append(Edge(source=str(u), target=str(v), label=edge_label))

    # Configure agraph with physics=False for a static layout
    config = Config(width="100%",
                    height=600,
                    directed=G_to_display.is_directed(),
                    physics=False,  # Key change: Set physics to False
                    hierarchical=False, # You can experiment with layout options
                    # Example layout options for static graphs (might need more tuning)
                    # layout={"randomSeed": 42}, # For consistent random layout
                    # Improved layout for static graph might involve pre-calculating positions
                    # and passing them to Node objects if streamlit-agraph supports it directly,
                    # or relying on its default static layout algorithm.
                    edges={"font": {"size": 10, "align": "top"}} # Edge label styling
                   )

    if nodes_vis or G_to_display.number_of_nodes() > 0: # Check if there are any nodes
        agraph(nodes=nodes_vis, edges=edges_vis, config=config)
    else:
        st.write("The current graph has no nodes to display.")
else:
    st.warning("No graph is available to display. Please generate a graph using the sidebar parameters.")
