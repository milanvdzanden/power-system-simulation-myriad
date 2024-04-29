
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
import time

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)
        
import power_system_simulation.graph_processing as pss       
def test_add():
    start_time = time.time()   
    
    # edge_id = 10
    vertex_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    edge_ids =             [1,      2,      3,      4,      5,      6,      7,      8,      9,       10]
    edge_vertex_id_pairs = [(2,6),  (1, 2), (2, 3), (3, 4), (6, 7), (6, 9), (4, 5), (7, 8), (9, 10), (1, 11)]
    edge_enabled =         [True,   True,   True,   True,   True,   True,   True,   False,  True,    True]
    source_vertex_id = 1
    gp = pss.GraphProcessor(vertex_ids, edge_ids,edge_vertex_id_pairs,edge_enabled,source_vertex_id)
    
    assert set(gp.find_downstream_vertices(1)) == set([6, 7, 9, 10])
    assert set(gp.find_downstream_vertices(2)) == set([2, 3, 4, 5, 6, 7, 9, 10])
    assert set(gp.find_downstream_vertices(3)) == set([3, 4, 5])
    assert set(gp.find_downstream_vertices(4)) == set([4, 5])
    assert set(gp.find_downstream_vertices(5)) == set([7])
    assert set(gp.find_downstream_vertices(6)) == set([9, 10])
    assert set(gp.find_downstream_vertices(7)) == set([5])
    assert set(gp.find_downstream_vertices(8)) == set([]) # Edge 8 is disabled
    assert set(gp.find_downstream_vertices(9)) == set([10])
    assert set(gp.find_downstream_vertices(10)) == set([11])
    
    """
    v1 ----- edge2 ------ v2 ----- edge3 ------ v3 ----- edge4 ------ v4 ----- edge7 ------ v5
    |                      |
  edge10                 edge1
    |                      |                          (disabled)
   v11                    v6 ----- edge5 ------ v7 ----- edge8 ------ v8
                           |
                         edge6
                           |
                          v9 ----- edge9 ------ v10
    """
    
    # print(gp.find_downstream_vertices(edge_id))
    print("Process finished --- %s seconds ---" % (time.time() - start_time))

test_add()

