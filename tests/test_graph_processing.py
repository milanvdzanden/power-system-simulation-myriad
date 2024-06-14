import os
import sys

import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

import networkx as nx

import power_system_simulation.graph_processing as pss


def test_downstream_vertices():
    # edge_id = 10
    vertex_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    edge_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    edge_vertex_id_pairs = [
        (2, 6),
        (1, 2),
        (2, 3),
        (3, 4),
        (6, 7),
        (6, 9),
        (4, 5),
        (7, 8),
        (9, 10),
        (1, 11),
        (4, 8)
    ]
    edge_enabled = [True, True, True, True, True, True, True, False, True, True, True]
    source_vertex_id = 1
    gp = pss.GraphProcessor(
        vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id
    )
    with pytest.raises(pss.IDNotFoundError) as excinfo:
        gp.find_downstream_vertices(12)

    assert set(gp.find_downstream_vertices(1)) == set([6, 7, 9, 10])
    assert set(gp.find_downstream_vertices(2)) == set([2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert set(gp.find_downstream_vertices(3)) == set([3, 4, 5, 8])
    assert set(gp.find_downstream_vertices(4)) == set([4, 5, 8])
    assert set(gp.find_downstream_vertices(5)) == set([7])
    assert set(gp.find_downstream_vertices(6)) == set([9, 10])
    assert set(gp.find_downstream_vertices(7)) == set([5])
    assert set(gp.find_downstream_vertices(8)) == set([])  # Edge 8 is disabled
    assert set(gp.find_downstream_vertices(9)) == set([10])
    assert set(gp.find_downstream_vertices(10)) == set([11])

    """
    v1 ----- edge2 ------ v2 ----- edge3 ------ v3 ----- edge4 ------ v4 ----- edge7 ------ v5
    |                      |                                           |
  edge10                 edge1                                       edge 11
    |                      |                          (disabled)       |
   v11                    v6 ----- edge5 ------ v7 ----- edge8 ------ v8
                           |
                         edge6
                           |
                          v9 ----- edge9 ------ v10
    """


# Move later to src if needed
def test_graph_generator():
    # Pre-set seed for random graph generation; Node amount
    pre_seed = 1
    pre_nodes = 50
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
        pss.GraphProcessor(
            t_vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id
        )

    # Non-unique edge-ID
    t_edge_ids = [1, 3, 5, 7, 8, 8]
    with pytest.raises(pss.IDNotUniqueError, match=r".*T1"):
        pss.GraphProcessor(
            vertex_ids, t_edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id
        )

    # edge_enabled and edge_id of different length
    t_edge_enabled = [True, True, True, False, False, True, True]
    with pytest.raises(pss.InputLengthDoesNotMatchError, match=r".*T0"):
        pss.GraphProcessor(
            vertex_ids, edge_ids, edge_vertex_id_pairs, t_edge_enabled, source_vertex_id
        )

    # edge_vertex_id_pairs and edge_ids of different length
    t_edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6)]
    with pytest.raises(pss.InputLengthDoesNotMatchError, match=r".*T1"):
        pss.GraphProcessor(
            vertex_ids, edge_ids, t_edge_vertex_id_pairs, edge_enabled, source_vertex_id
        )

    # Non-valid edge_vertex_id_pairs
    t_edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 999), (2, 4), (4, 6), (2, 10)]
    with pytest.raises(pss.IDNotFoundError, match=r".*T0"):
        pss.GraphProcessor(
            vertex_ids, edge_ids, t_edge_vertex_id_pairs, edge_enabled, source_vertex_id
        )

    # Non-valid source_vertex_id
    t_source_vertex_id = 999
    with pytest.raises(pss.IDNotFoundError, match=r".*T1"):
        pss.GraphProcessor(
            vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, t_source_vertex_id
        )

    # Graph not fully connected
    t_vertex_ids = [0, 2, 4, 6, 10, 999]
    with pytest.raises(pss.GraphNotFullyConnectedError):
        pss.GraphProcessor(
            t_vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id
        )

    # Cyclic graph
    t_edge_enabled = [True, True, True, True, True, True]
    with pytest.raises(pss.GraphCycleError):
        pss.GraphProcessor(
            vertex_ids, edge_ids, edge_vertex_id_pairs, t_edge_enabled, source_vertex_id
        )


def test_alternative_edges():
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [1, 3, 5, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)]
    edge_enabled = [True, True, True, False, False, True]
    source_vertex_id = 0

    gp = pss.GraphProcessor(
        vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id
    )

    assert set(gp.find_alternative_edges(1)) == set([7])
    assert set(gp.find_alternative_edges(3)) == set([8, 7])
    assert set(gp.find_alternative_edges(5)) == set([8])
    assert set(gp.find_alternative_edges(9)) == set([])

    with pytest.raises(pss.IDNotFoundError) as excinfo:
        gp.find_alternative_edges(14)

    with pytest.raises(pss.EdgeAlreadyDisabledError) as excinfo:
        gp.find_alternative_edges(7)


test_graph_creation()
test_alternative_edges()
test_downstream_vertices()
