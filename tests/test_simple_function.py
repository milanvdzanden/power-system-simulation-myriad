from power_system_simulation.simple_function import add
import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_edge(1,2)
G.add_edge(2,3,id=23)
nx.draw(G, with_labels = True)
plt.show()
print(list(G.edges(data=True))[1][2].get('id'))
# print(G.edges(data=True))


# vertex_ids = list(G.nodes(data=True))[],
# edge_ids = list(G.edges(data=True))[][].get('id'),
# edge_vertex_id_pairs = list(G.edges(data=True))[][],
# edge_enabled: List[bool],
# source_vertex_id: int,
        
        
def test_add():
    assert add(1, 1) == 2

