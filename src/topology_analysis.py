"""
topology_analysis.py
--------------------
Build a PPI graph from a preprocessed edge list and compute topological metrics:
degree, clustering coefficient, and shortest paths.
"""

import pandas as pd
import networkx as nx


def build_graph(df: pd.DataFrame, source_col: str = "protein1", target_col: str = "protein2") -> nx.Graph:
    """Construct an undirected graph from a DataFrame of edges."""
    G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)
    return G


def compute_degree(G: nx.Graph) -> pd.DataFrame:
    """Return a DataFrame with each node and its degree."""
    degrees = dict(G.degree())
    return pd.DataFrame(list(degrees.items()), columns=["protein", "degree"]).sort_values(
        "degree", ascending=False
    ).reset_index(drop=True)


def compute_clustering(G: nx.Graph) -> pd.DataFrame:
    """Return a DataFrame with each node and its clustering coefficient."""
    clustering = nx.clustering(G)
    return pd.DataFrame(list(clustering.items()), columns=["protein", "clustering_coefficient"]).sort_values(
        "clustering_coefficient", ascending=False
    ).reset_index(drop=True)


def compute_shortest_paths(G: nx.Graph) -> dict:
    """
    Compute all-pairs shortest path lengths for the largest connected component.
    Returns a dict-of-dicts: {source: {target: length}}.
    Returns an empty dict if the graph has no nodes.
    """
    if G.number_of_nodes() == 0:
        return {}
    largest_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return dict(nx.all_pairs_shortest_path_length(largest_cc))


def run_topology_analysis(
    edge_csv: str,
    output_degree: str,
    output_clustering: str,
    source_col: str = "protein1",
    target_col: str = "protein2",
) -> nx.Graph:
    """End-to-end topology analysis: build graph, compute metrics, and save CSVs."""
    df = pd.read_csv(edge_csv)
    G = build_graph(df, source_col=source_col, target_col=target_col)

    degree_df = compute_degree(G)
    degree_df.to_csv(output_degree, index=False)
    print(f"Degree distribution saved to {output_degree}")

    clustering_df = compute_clustering(G)
    clustering_df.to_csv(output_clustering, index=False)
    print(f"Clustering coefficients saved to {output_clustering}")

    return G


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python topology_analysis.py <edge_csv> <output_degree_csv> <output_clustering_csv>")
        sys.exit(1)

    run_topology_analysis(sys.argv[1], sys.argv[2], sys.argv[3])
