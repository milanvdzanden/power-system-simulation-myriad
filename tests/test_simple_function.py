import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)

import power_system_simulation.graph_processing as pss

def test_altedges():
    
    vertex_ids = [0,1]
    edge_ids = [20]
    edge_vertex_id_pairs = [(0,1)]
    edge_enabled = [True]
    source_vertex_id = 0
    gp = pss.GraphProcessor(vertex_ids, edge_ids,edge_vertex_id_pairs,edge_enabled,source_vertex_id);
    gp.find_alternative_edges(20);

test_altedges();
  
