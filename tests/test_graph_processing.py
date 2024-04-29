import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)

import power_system_simulation.graph_processing as pss
import networkx as nx

def test_graph_processing():  
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [1, 3, 5, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)]
    edge_enabled = [True, True, True, False, False, True]
    source_vertex_id = 0
    gp = pss.GraphProcessor(vertex_ids, edge_ids,edge_vertex_id_pairs,edge_enabled,source_vertex_id)
    
    assert set(gp.find_alternative_edges(1)) == set([7])
    assert set(gp.find_alternative_edges(3)) == set([8, 7])
    assert set(gp.find_alternative_edges(5)) == set([8])
    assert set(gp.find_alternative_edges(9)) == set([])

test_graph_processing();