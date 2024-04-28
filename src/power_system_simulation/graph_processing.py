"""
This is a skeleton for the graph processing assignment.

We define a graph processor class with some function skeletons.
"""


import networkx as nx
from typing import List, Tuple



class IDNotFoundError(Exception):
    pass


class InputLengthDoesNotMatchError(Exception):
    pass


class IDNotUniqueError(Exception):
    pass


class GraphNotFullyConnectedError(Exception):
    pass


class GraphCycleError(Exception):
    pass


class EdgeAlreadyDisabledError(Exception):
    pass


class GraphProcessor:
    """
    General documentation of this class.
    You need to describe the purpose of this class and the functions in it.
    We are using an undirected graph in the processor.
    """

    def __init__(
        self,
        vertex_ids: List[int],
        edge_ids: List[int],
        edge_vertex_id_pairs: List[Tuple[int, int]],
        edge_enabled: List[bool],
        source_vertex_id: int,
    ) -> None:
        """
        Initialize a graph processor object with an undirected graph.
        Only the edges which are enabled are taken into account.
        Check if the input is valid and raise exceptions if not.
        The following conditions should be checked:
            1. vertex_ids and edge_ids should be unique. (IDNotUniqueError)
            2. edge_vertex_id_pairs should have the same length as edge_ids. (InputLengthDoesNotMatchError)
            3. edge_vertex_id_pairs should contain valid vertex ids. (IDNotFoundError)
            4. edge_enabled should have the same length as edge_ids. (InputLengthDoesNotMatchError)
            5. source_vertex_id should be a valid vertex id. (IDNotFoundError)
            6. The graph should be fully connected. (GraphNotFullyConnectedError)
            7. The graph should not contain cycles. (GraphCycleError)
        If one certain condition is not satisfied, the error in the parentheses should be raised.

        Args:
            vertex_ids: list of vertex ids
            edge_ids: liest of edge ids
            edge_vertex_id_pairs: list of tuples of two integer
                Each tuple is a vertex id pair of the edge.
            edge_enabled: list of bools indicating of an edge is enabled or not
            source_vertex_id: vertex id of the source in the graph
        """
        # put your implementation here
        self.vertex_ids = vertex_ids
        self.edge_ids = edge_ids
        self.edge_vertex_id_pairs = edge_vertex_id_pairs
        self.edge_enabled = edge_enabled
        self.source_vertex_id = source_vertex_id
        pass
    
    def find_downstream_vertices(self, edge_id: int) -> List[int]:
        """
        Given an edge id, return all the vertices which are in the downstream of the edge,
            with respect to the source vertex.
            Including the downstream vertex of the edge itself!

        Only enabled edges should be taken into account in the analysis.
        If the given edge_id is a disabled edge, it should return empty list.
        If the given edge_id does not exist, it should raise IDNotFoundError.


        For example, given the following graph (all edges enabled):

            vertex_0 (source) --edge_1-- vertex_2 --edge_3-- vertex_4

        Call find_downstream_vertices with edge_id=1 will return [2, 4]
        Call find_downstream_vertices with edge_id=3 will return [4]

        Args:
            edge_id: edge id to be searched

        Returns:
            A list of all downstream vertices.
        """
        vertex_ids = self.vertex_ids
        edge_ids = self.edge_ids 
        edge_vertex_id_pairs =  self.edge_vertex_id_pairs
        edge_enabled = self.edge_enabled 
        source_vertex_id = self.source_vertex_id 
        output = []

        for x in range(len(edge_ids)):                      # Problem with disabled edges, indexes chaning places
            if edge_ids[x] == edge_id:
                input_index_edge = x
                break
        if edge_enabled[input_index_edge] == False:
            return output
        
# List for storing all enabled edges, to use for finding out if the graph is fully connected
        edge_vertex_id_pairs_enabled = []   
        edge_ids_enabled = []
        
        # For loop for finding all enabled edges
        for edge_to_check in edge_vertex_id_pairs:          # Problem with disabled edges, indexes chaning places
            edge_to_check_index = edge_vertex_id_pairs.index(edge_to_check)
            
            if edge_enabled[edge_to_check_index] == True:
                edge_ids_enabled.append(edge_to_check)
                edge_vertex_id_pairs_enabled.append(edge_to_check)
                
        for x in range(len(edge_ids_enabled)):              # Problem with disabled edges, indexes chaning places
            if edge_ids_enabled[x] == edge_id:
                input_index_edge = x
                break
        
        G = nx.Graph()
        G.add_nodes_from(vertex_ids)
        G.add_edges_from(edge_vertex_id_pairs_enabled)

# Determining shortest length to source
        path_1 = nx.shortest_path(G, source_vertex_id, edge_vertex_id_pairs_enabled[input_index_edge][0] , weight=None)    
        path_2 = nx.shortest_path(G, source_vertex_id, edge_vertex_id_pairs_enabled[input_index_edge][1] , weight=None)       

        if len(path_1) < len(path_2):
            shortest_path = path_1
            shortest_index = 1
        else: 
            shortest_path = path_2
            shortest_index = 0
            
#Defines all paths from source to leaf nodes   
        paths_to_leafs = []
        stack = [(source_vertex_id, [source_vertex_id])]

        while stack:
            current_node, path = stack.pop()
            if G.degree(current_node) == 1:  # Check for leaf node
                paths_to_leafs.append(path)
            else:
                for neighbor in G.neighbors(current_node):
                    if neighbor not in path:  # Avoid revisiting nodes
                        stack.append((neighbor, path + [neighbor]))
        
        print(edge_vertex_id_pairs_enabled[input_index_edge])
        # print(paths_to_leafs[:])
        paths_to_leafs_including_input = []
        for x in paths_to_leafs:
            print(x)
            if edge_vertex_id_pairs_enabled[input_index_edge][shortest_index] in x:
                for y in x:
                    print(y)
                    if y not in paths_to_leafs_including_input:
                        paths_to_leafs_including_input.append(y)
                        
        # print(paths_to_leafs_including_input)
        # print(shortest_path)
        output = list(set(paths_to_leafs_including_input) - set(shortest_path))
        print(output)
        return output

    def find_alternative_edges(self, disabled_edge_id: int) -> List[int]:
        """
        Given an enabled edge, do the following analysis:
            If the edge is going to be disabled,
                which (currently disabled) edge can be enabled to ensure
                that the graph is again fully connected and acyclic?
            Return a list of all alternative edges.
        If the disabled_edge_id is not a valid edge id, it should raise IDNotFoundError.
        If the disabled_edge_id is already disabled, it should raise EdgeAlreadyDisabledError.
        If there are no alternative to make the graph fully connected again, it should return empty list.


        For example, given the following graph:

        vertex_0 (source) --edge_1(enabled)-- vertex_2 --edge_9(enabled)-- vertex_10
                 |                               |
                 |                           edge_7(disabled)
                 |                               |
                 -----------edge_3(enabled)-- vertex_4
                 |                               |
                 |                           edge_8(disabled)
                 |                               |
                 -----------edge_5(enabled)-- vertex_6

        Call find_alternative_edges with disabled_edge_id=1 will return [7]
        Call find_alternative_edges with disabled_edge_id=3 will return [7, 8]
        Call find_alternative_edges with disabled_edge_id=5 will return [8]
        Call find_alternative_edges with disabled_edge_id=9 will return []

        Args:
            disabled_edge_id: edge id (which is currently enabled) to be disabled

        Returns:
            A list of alternative edge ids.
        """
        # put your implementation here
        pass

