import os
import sys

import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

import networkx as nx

import power_system_simulation.graph_processing as pss


# Move later to src if needed
def test_graph_generator():
    # Pre-set seed for random graph generation; Node amount
    pre_seed = 1
    pre_nodes = 5
    # Create graph and assign source node
    graph = nx.random_labeled_rooted_tree(pre_nodes, seed=pre_seed)
    leaf_nodes = [node for node in graph.nodes() if graph.degree(node) == 1]
    # Any leaf node is a valid source node, so assign to the first one
    source_node = leaf_nodes[0]
    return graph, source_node


def test_graph_creation():
    # Working graph
    graph, source = test_graph_generator()
    pss.GraphProcessor(
        list(graph.nodes),
        [x for x in range(0, len(list(graph.edges)))],
        list(graph.edges),
        [True for x in range(0, len(graph.edges))],
        source,
    )

    # Working graph (cyclic though)
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [1, 3, 5, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)]
    edge_enabled = [True, True, True, False, False, True]
    source_vertex_id = 0

    # Non-unique vertex-ID
    t_vertex_ids = [0, 2, 4, 6, 6]
    with pytest.raises(pss.IDNotUniqueError, match=r".*T0"):
        pss.GraphProcessor(t_vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)

    # Non-unique edge-ID
    t_edge_ids = [1, 3, 5, 7, 8, 8]
    with pytest.raises(pss.IDNotUniqueError, match=r".*T1"):
        pss.GraphProcessor(vertex_ids, t_edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)

    # edge_enabled and edge_id of different length
    t_edge_enabled = [True, True, True, False, False, True, True]
    with pytest.raises(pss.InputLengthDoesNotMatchError, match=r".*T0"):
        pss.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, t_edge_enabled, source_vertex_id)

    # edge_vertex_id_pairs and edge_ids of different length
    t_edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6)]
    with pytest.raises(pss.InputLengthDoesNotMatchError, match=r".*T1"):
        pss.GraphProcessor(vertex_ids, edge_ids, t_edge_vertex_id_pairs, edge_enabled, source_vertex_id)

    # Non-valid edge_vertex_id_pairs
    t_edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 999), (2, 4), (4, 6), (2, 10)]
    with pytest.raises(pss.IDNotFoundError, match=r".*T0"):
        pss.GraphProcessor(vertex_ids, edge_ids, t_edge_vertex_id_pairs, edge_enabled, source_vertex_id)

    # Non-valid source_vertex_id
    t_source_vertex_id = 999
    with pytest.raises(pss.IDNotFoundError, match=r".*T1"):
        pss.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, t_source_vertex_id)

    # Graph not fully connected
    t_vertex_ids = [0, 2, 4, 6, 10, 999]
    with pytest.raises(pss.GraphNotFullyConnectedError):
        pss.GraphProcessor(t_vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)

    # Cyclic graph
    t_edge_enabled = [True, True, True, True, True, True]
    with pytest.raises(pss.GraphCycleError):
        pss.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, t_edge_enabled, source_vertex_id)


def test_graph_processing():
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [1, 3, 5, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)]
    edge_enabled = [True, True, True, False, False, True]
    source_vertex_id = 0
    # gp = pss.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
    # graph, source = test_graph_generator()
    # gp = pss.GraphProcessor(
    #    list(graph.nodes),
    #    [x for x in range(0, len(list(graph.edges)))],
    #    list(graph.edges),
    #    [True for x in range(0, len(graph.nodes) - 1)],
    #    source,
    # )


test_graph_creation()
test_graph_processing()
