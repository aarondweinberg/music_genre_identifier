# This script contains functions to generate a graph from the musicmap website,
# produce an adjacency matrix, and create a tensor of shortest paths

# Imports
import os
import torch
import numpy as np
import pandas as pd
import networkx as nx
from bs4 import BeautifulSoup
from collections import defaultdict

# A function to standardize the names of the various nodes in the musicmap graph
# This is used in the create_musicmap_graph function
def normalize_node(node):
    """
    Standardizes the musicmap genre labels

    Args:
        A node in the musicmap graph

    Returns:
        The 'cleaned' name of the node
    """
    # Fix static
    if node == 'static1': node = 'sta01'
    elif node == 'static01': node = 'sta01'
    elif node == 'static2': node = 'sta02'
    elif node == 'static02': node = 'sta02'
    elif node == 'static03': node = 'sta03'
    elif node == 'static3': node = 'sta03'

    # Fix static-genre
    static_genre_map = {
        'worldA': 'woA',
        'worldB': 'woB',
        'worldC': 'woC',
        'worldA9': 'woA09', 
        'worldA09': 'woA09', 
        'worldA13': 'woA13', 
        'worldB2': 'woB02', 
        'worldB02': 'woB02', 
        'worldB4': 'woB04',
        'worldB04': 'woB04',
        'worldB5': 'woB05', 
        'worldB05': 'woB05', 
        'worldC7': 'woC07', 
        'worldC07': 'woC07', 
        'worldC8': 'woC08',
        'worldC08': 'woC08',
        'worldC10': 'woC10',
        'worldC13': 'woC13',
        'worldC14': 'woC14',
        'util3': 'uti03',
        'util01': 'uti01',
        'util02': 'uti02',
        'util03': 'uti03'
    }
    node = static_genre_map.get(node, node)

    # Fix inl/inr prefix
    if node.startswith("inl") or node.startswith("inr"):
        node = "ind" + node[3:]

    return node


# A function to create the musicmap graph object G
# It has default values for the edge weights
def create_musicmap_graph(
    primary_edge_weight = 1, 
    secondary_edge_weight = 2,
    backlash_edge_weight = 3,
    supergenre_edge_weight = 4,
    world_supergenre_weight = 1,
    world_cluster_weight = 1,
    util_genre_weight = 2,
    cluster_weight = 1
):

    """
    Creates a NetworkX graph G to represent the musicmap graph

    Args:
        A collection of eight edge weights; default values are specified above

    Returns:
        G: a NetworkX graph
    """

    # Step 1 - Load the CSV
    df = pd.read_csv("musicmap_genres.csv")

    # Step 2 - Create graph and type-based lists
    G = nx.Graph()
    type_lists = defaultdict(list)

    for _, row in df.iterrows():
        code = normalize_node(row['Code'])
        typ = row['Type']
        G.add_node(code, type=typ)
        type_lists[typ].append(code)

    # Fix static, static-genre, and inl/inr node dames
    #G = nx.relabel_nodes(G, normalize_node)
    G = nx.relabel_nodes(G, lambda x: normalize_node(x))

    # Ensure all supergenres are present as nodes
    supergenres_in_csv = set(df['Supergenre'].dropna())
    for supergenre in supergenres_in_csv:
        if supergenre not in G:
            G.add_node(supergenre, type="supergenre")

    # Load HTML
    with open("Musicmap.html", encoding="utf-8") as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Extract_edges function with normalization
    def extract_edges(group_class, weight):
        g = soup.find("g", class_=group_class)
        if g:
            for path in g.find_all("path"):
                cls = path.get("class")
                if isinstance(cls, list):
                    parts = cls
                elif isinstance(cls, str):
                    parts = cls.strip().split()
                else:
                    continue  # Skip malformed entries

                if len(parts) == 2:
                    node1 = normalize_node(parts[0])
                    node2 = normalize_node(parts[1])
                    G.add_edge(node1, node2, weight=weight)


    # Add edges
    extract_edges("primary", primary_edge_weight)
    extract_edges("secondary", secondary_edge_weight)
    extract_edges("backlash", backlash_edge_weight)

    # Genre to super-genre connections
    for _, row in df.iterrows():
        if row["Type"] == "genre":
            genre = row["Code"]
            supergenre = row["Supergenre"]
            if genre in G and supergenre in G:
                G.add_edge(genre, supergenre, weight=supergenre_edge_weight)

    # Clusters
    cluster_edges = {
        "rock": {"alt", "con", "gld", "hcp", "pwv", "rnr"},
        "blue_note": {"blu", "gos", "jaz"},
        "breakbeat_dance": {"brb", "dnb", "hct"},
        "four_to_the_floor": {"hct", "hou", "tec", "tra"},
        "edm": {"brb", "dnb", "hct", "hou", "tec", "tra"},
        "heavy": {"hcp", "ind", "met", "pwv"},
        "poprock": {"alt", "con", "gld", "hcp", "pwv", "rnr", "cou", "ind", "met", "pop"},
        "rhythm": {"blu", "gos", "jaz", "jam", "rap", "rnb"},
        "electronic": {"brb", "dnb", "hct", "hou", "tec", "tra", "dtp", "ind", "rap"},
    }
    for cluster, nodes in cluster_edges.items():
        for node in nodes:
            G.add_edge(cluster, node, weight=cluster_weight)

    # Connect static-genre → static-subgroup
    for node in type_lists["static-genre"]:
        if node.startswith(("woA", "woB", "woC")):
            subgroup = node[:-1]  # crude match; adjust as needed
            matches = [s for s in type_lists.get("static-subgroup", []) if s.startswith(node[:6])]
            for match in matches:
                G.add_edge(node, match, weight=world_supergenre_weight)

    # Connect - static-subgroup → world
    for node in type_lists.get("static-subgroup", []):
        G.add_edge(node, "world", weight=world_cluster_weight)

    # Connect - util* → util
    for node in type_lists["static-genre"]:
        if node.startswith("util"):
            G.add_edge(node, "util", weight=util_genre_weight)
    
    return G

# A function to load class names from the base directory to generate the shortest path tensor
def load_class_names_from_directory(base_dir):
    """Return sorted list of class folder names."""
    return sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

# A function to compute the matrix of shortest distances from a given graph
def compute_shortest_paths_between_classes_nx(class_dir, graph, return_tensor=True):
    """
    Given a directory of class folders and a NetworkX graph (with extra intermediate nodes),
    compute shortest path lengths between just the leaf-level class nodes.
    """
    # Step 1: Load class names from folders
    class_names = load_class_names_from_directory(class_dir)

    # Step 2: Verify that all class names are in the graph
    missing = [cls for cls in class_names if cls not in graph.nodes]
    if missing:
        raise ValueError(f"These class folders are missing from the graph nodes: {missing}")

    # Step 3: Compute full all-pairs shortest paths using Floyd-Warshall
    # Result is a dict-of-dict structure: {source: {target: distance}}
    shortest_paths = dict(nx.floyd_warshall(graph))

    # Step 4: Build submatrix for leaf class distances
    n = len(class_names)
    dist_matrix = np.zeros((n, n), dtype=np.float32)

    for i, src in enumerate(class_names):
        for j, tgt in enumerate(class_names):
            dist = shortest_paths[src].get(tgt, np.inf)
            dist_matrix[i, j] = dist

    if return_tensor:
        return torch.tensor(dist_matrix, dtype=torch.float32), class_names
    else:
        return dist_matrix, class_names