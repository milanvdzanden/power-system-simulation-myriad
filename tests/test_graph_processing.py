import os
import sys
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

import networkx as nx
import power_system_simulation.graph_processing as pss


def test_graph_processing():
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [1, 3, 5, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)]
    edge_enabled = [True, True, True, False, False, True]
    source_vertex_id = 0
    gp = pss.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)

    assert set(gp.find_alternative_edges(1)) == set([7])
    assert set(gp.find_alternative_edges(3)) == set([8, 7])
    assert set(gp.find_alternative_edges(5)) == set([8])
    assert set(gp.find_alternative_edges(9)) == set([])

    with pytest.raises(pss.IDNotFoundError) as excinfo:
        gp.find_alternative_edges(14)

    with pytest.raises(pss.EdgeAlreadyDisabledError) as excinfo:
        gp.find_alternative_edges(7)


def test_input_length_does_not_match_error():
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [1, 3, 5, 7, 8]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10), (6, 10)]  # Adding one extra pair
    edge_enabled = [True, True, True, False, False, True]
    source_vertex_id = 0

    with pytest.raises(pss.InputLengthDoesNotMatchError) as excinfo:
        pss.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)


def test_id_not_unique_error():
    vertex_ids = [0, 2, 4, 6, 10, 10]  # Duplicate vertex id
    edge_ids = [1, 3, 5, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)]
    edge_enabled = [True, True, True, False, False, True]
    source_vertex_id = 0

    with pytest.raises(pss.IDNotUniqueError) as excinfo:
        pss.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)


def test_graph_not_fully_connected_error():
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [1, 3, 5, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (2, 4), (4, 6), (2, 10)]  # Missing edges connected to vertex 0
    edge_enabled = [True, True, True, False, False, True]
    source_vertex_id = 0

    with pytest.raises(pss.GraphNotFullyConnectedError) as excinfo:
        pss.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)


def test_graph_cycle_error():
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [1, 3, 5, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (2, 4), (4, 6), (6, 2), (2, 10)]  # Introducing a cycle
    edge_enabled = [True, True, True, False, False, True]
    source_vertex_id = 0

    with pytest.raises(pss.GraphCycleError) as excinfo:
        pss.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)


test_graph_processing()
test_input_length_does_not_match_error()
test_id_not_unique_error()
test_graph_not_fully_connected_error()
test_graph_cycle_error()
