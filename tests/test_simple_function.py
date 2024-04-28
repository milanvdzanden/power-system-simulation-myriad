
# import networkx as nx
# import matplotlib.pyplot as plt
# G = nx.Graph()
# G.add_node(1)
# G.add_node(2)
# G.add_node(3)
# G.add_edge(1,2)
# G.add_edge(2,3,id=23)
# nx.draw(G, with_labels = True)
# plt.show()
# print(list(G.edges(data=True))[1][2].get('id'))
# print(G.edges(data=True))


# vertex_ids = list(G.nodes(data=True))[],
# edge_ids = list(G.edges(data=True))[][].get('id'),
# edge_vertex_id_pairs = list(G.edges(data=True))[][],
# edge_enabled: List[bool],
# source_vertex_id: int,
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)
        
import power_system_simulation.graph_processing as pss       
def test_add():
    edge_id = 1
    edge_ids = [0,1,2,3,4]
    vertex_ids = [10,11,12,13,14,15]
    edge_vertex_id_pairs = [[11,12],[12,13],[13,14],[14,10],[10,15]]
    edge_enabled = [True,True,True,True,True]
    source_vertex_id = 11
    gp = pss.GraphProcessor(vertex_ids, edge_ids,edge_vertex_id_pairs,edge_enabled,source_vertex_id)
    

    gp.find_downstream_vertices(edge_id)

test_add()

