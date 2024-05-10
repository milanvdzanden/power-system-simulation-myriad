"""
Graph processing file, handling the low level graph calculations
for finding alternative edges and downstream vertices
We define a graph processor class with some function skeletons.
"""

from typing import List, Tuple

import networkx as nx


class IDNotFoundError(Exception):
    """
    Error class for IDNotFoundError
    """

    def __init__(self, mode):
        """
        Prints the exception message using the standard exception class.

        Args:
          self: passes exception
          mode: error type,
            0 if vertex pair does not exist in vertex_ids
            1 if source_vertex_id does not exist in vertex_ids
        """
        Exception.__init__(
            self,
            "IDNotFoundError: vertex pair does not exist in vertex_ids (if 0)"
            + "or source_vertex_id does not exist in vertex_ids (if 1): T"
            + str(mode),
        )


class InputLengthDoesNotMatchError(Exception):
    """
    Error class for InputLenghtDoesNotMatchError
    """

    def __init__(self, mode, l_primary_length, l_edge_ids):
        """
        Prints the exception message using the standard exception class.

        Args:
          self: passes exception
          mode: error type,
            0 if the length of edge_enabled is different than edge_ids
            1 if the length of edge_vertex_id_pairs is different than edge_ids
          l_primary_length: length of non-matching list,
            edge_enabled if mode is 0
            edge_vertex_id_pairs if mode is 1
          l_edge_ids: length of edge_ids list
        """
        Exception.__init__(
            self,
            "InputLengthDoesNotMatchError:"
            + "The length of edge_enabled"
            + "(if 0) or edge_vertex_id_pairs (if 1) ("
            + str(l_primary_length)
            + ") is different than edge_ids ("
            + str(l_edge_ids)
            + "): T"
            + str(mode),
        )


class IDNotUniqueError(Exception):
    """
    Error class for class IDNotUniqueError
    """

    def __init__(self, mode):
        """
        Prints the exception message using the standard exception class.

        Args:
          self: passes exception
          mode: error type,
            0 if there are non-unique vertices in the graph
            1 if there are non-unique edges in the graph
        """
        Exception.__init__(
            self,
            "IDNotUniqueError: There are non-unique vertices (if 0)"
            + "or edges (if 1) in the graph: T"
            + str(mode),
        )
        
class GraphNotFullyConnectedError(Exception):
    """
    Error class for GraphNotFullyConnectedError
    """

    def __init__(self):
        """
        Prints the exception message using the standard exception class.

        Args:
          self: passes exception
        """
        Exception.__init__(self, "The graph is not fully connected.")


class GraphCycleError(Exception):
    """
    Error class for GraphCycleError
    """

    def __init__(self):
        """
        Prints the exception message using the standard exception class.

        Args:
          self: passes exception
        """
        Exception.__init__(self, "The graph contains cycles.")


class EdgeAlreadyDisabledError(Exception):
    """
    Error class for EdgeAlreadyDisabledError
    """

    def __init__(self, edge_disabled_id):
        """
        Prints the exception message using the standard exception class.

        Args:
          self: passes exception
          edge_disabled_id: the id of the edge that is already disabled
        """
        Exception.__init__(self, "The edge is already disabled: " + str(edge_disabled_id))


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
            (IDNotUniqueError)
            1. vertex_ids and edge_ids should be unique.
            (InputLengthDoesNotMatchError)
            2. edge_vertex_id_pairs should have the same length as edge_ids.
            (IDNotFoundError)
            3. edge_vertex_id_pairs should contain valid vertex ids.
            (InputLengthDoesNotMatchError)
            4. edge_enabled should have the same length as edge_ids.
            (IDNotFoundError)
            5. source_vertex_id should be a valid vertex id.
            (GraphNotFullyConnectedError)
            6. The graph should be fully connected.
            (GraphCycleError)
            7. The graph should not contain cycles.
        If one certain condition is not satisfied, the error
        in the parentheses should be raised.

        Args:
            vertex_ids: list of vertex ids
            edge_ids: liest of edge ids
            edge_vertex_id_pairs: list of tuples of two integer
                Each tuple is a vertex id pair of the edge.
            edge_enabled: list of bools indicating of an edge is enabled or not
            source_vertex_id: vertex id of the source in the graph
        """
        # Check: vertex_ids - is unique?
        if not len(vertex_ids) == len(set(vertex_ids)):
            raise IDNotUniqueError(0)
        
        # Check: edge_ids - is unique?
        if not len(edge_ids) == len(set(edge_ids)):
            raise IDNotUniqueError(1)

        # Check: edge_enabled and edge_ids - are same length?
        if not len(edge_enabled) == len(edge_ids):
            raise InputLengthDoesNotMatchError(0, len(edge_enabled), len(edge_ids))
        
        # Check: edge_vertex_id_pairs and edge_ids - are same length?
        if not len(edge_vertex_id_pairs) == len(edge_ids):
            raise InputLengthDoesNotMatchError(1, len(edge_vertex_id_pairs), len(edge_ids))

        # Check: edge_vertex_id_pairs - are vertex ids valid?
        for x in edge_vertex_id_pairs:
            if (x[0] not in vertex_ids) or (x[1] not in vertex_ids):
                raise IDNotFoundError(0)
            
        # Check: source_vertex_id - is source vortex id valid?
        if source_vertex_id not in vertex_ids:
            raise IDNotFoundError(1)

        # Basic checks completed, graph can now be constructed.
        self.graph_enabled_edges = nx.Graph()
        self.graph_enabled_edges_ids = [];
        self.graph_enabled_edges.add_nodes_from(vertex_ids)
        edge_vertex_id_pairs_enabled = []
        
        # Find the list of enabled edges
        for x in range(0, len(edge_vertex_id_pairs)):
            if edge_enabled[x]:
                self.graph_enabled_edges_ids.append(edge_ids[x])
                edge_vertex_id_pairs_enabled.append(edge_vertex_id_pairs[x])
                
        self.graph_enabled_edges.add_edges_from(edge_vertex_id_pairs_enabled)
        self.graph_all_edges = nx.Graph()
        self.graph_all_edges.add_nodes_from(vertex_ids)
        self.graph_all_edges.add_edges_from(edge_vertex_id_pairs)
        
        # Check: graph - is fully connected?
        if not nx.is_connected(self.graph_all_edges):
            raise GraphNotFullyConnectedError()
        
        # Check: graph - has no cycles?
        try:
            nx.find_cycle(self.graph_enabled_edges)
        except nx.NetworkXNoCycle:
            pass
        else:
            raise GraphCycleError()

        self.vertex_ids = vertex_ids
        self.edge_ids = edge_ids
        self.edge_vertex_id_pairs = edge_vertex_id_pairs
        self.edge_enabled = edge_enabled
        self.source_vertex_id = source_vertex_id

    def find_downstream_vertices(self, edge_id: int) -> List[int]:
        output = []

        # Check if disabled_edge_id exists
        if edge_id not in self.edge_ids:
            raise IDNotFoundError(1)

        # Calculate the index of the input edge
        input_index_edge = self.edge_ids.index(edge_id)

        # If the input edge is already disabled, return empty set
        if self.edge_enabled[input_index_edge] is False:
            return output

        # Step 1: Calculate the islands and store them
        # Step 2: Remove the input edge from the id_pairs list
        # Step 3: Calculate the new islands
        # Step 4: Remove the islands that were already there in Step 1
        # Step 5: Check which remaining islands contain the source vertex, the other one
        # IS the output

        # List for storing all enabled edges, to use for finding out if the graph is fully connected
        edge_vertex_id_pairs_enabled_before = []
        edge_ids_enabled_before = []

        # For loop for finding all enabled edges
        for edge_to_check in self.edge_vertex_id_pairs:
            edge_to_check_index = self.edge_vertex_id_pairs.index(edge_to_check)

            if self.edge_enabled[edge_to_check_index] is True:
                edge_ids_enabled_before.append(self.edge_ids[edge_to_check_index])
                edge_vertex_id_pairs_enabled_before.append(edge_to_check)

        # Step 1 is calculated here
        ## Island calculation before removing the input edge
        graph = nx.Graph()
        graph.add_nodes_from(self.vertex_ids)
        graph.add_edges_from(edge_vertex_id_pairs_enabled_before)

        list_of_islands_before = []

        for i,c in enumerate(nx.connected_components(graph)):
            list_of_islands_before.append(list(c))

        # Step 2 is calculated here
        # For loop for finding all enabled edges without the input edge
        edge_vertex_id_pairs_enabled_after = []
        edge_ids_enabled_after = []  # The list with the input edge removed
        for edge_index in range(len(edge_ids_enabled_before)):
            if edge_ids_enabled_before[edge_index] != edge_id:
                edge_ids_enabled_after.append(edge_ids_enabled_before[edge_index])
                edge_vertex_id_pairs_enabled_after.append(
                    edge_vertex_id_pairs_enabled_before[edge_index]
                )

        # Step 3: Calculate the new islands
        ## Island calculation after
        graph = nx.Graph()
        graph.add_nodes_from(self.vertex_ids)
        graph.add_edges_from(edge_vertex_id_pairs_enabled_after)

        list_of_islands_after = []

        for i,c in enumerate(nx.connected_components(graph)):
            list_of_islands_after.append(list(c))

        output_list = []

        # Step 4: Remove the islands that were already there in Step 1
        # Step 5: Check which remaining islands contain the source vertex, the other one
        # IS the output
        # We loop through all the islands, if an island contains the source vertex it is removed,
        # and if the island already existed initially it is also removed.
        # What you are left with is a list of the downstream vertices
        for id_list in list_of_islands_after:
            contains_source = False
            if id_list in list_of_islands_before:
                continue
            for j in id_list:
                if j is self.source_vertex_id:
                    contains_source = True
            if not contains_source:
                output_list.append(id_list)

        output = output_list[0]
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
        If there are no alternative to make the graph fully connected again,
        it should return empty list.

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
        # Ouput variable list
        output = []

        # Check if disabled_edge_id exists
        if disabled_edge_id not in self.edge_ids:
            raise IDNotFoundError(0)

        # Check if the edge is already disabled with and index
        edge_index = self.edge_ids.index(disabled_edge_id)

        # Check if disabled_edge_id is already disabled
        if not self.edge_enabled[edge_index]:
            raise EdgeAlreadyDisabledError(1)

        # Loop through all disabled edges
        # Enable that edge temporarily and check if it is again fully connected
        # Check if it is still non-circular
        # If all of this passed, the edge is an alternative edge

        # Loop through ALL edges (both enabled and disabled)
        for edge in self.edge_ids:
            current_edge_index = self.edge_ids.index(edge)

            # Check if the edge is disabled, if so, continue
            if self.edge_enabled[current_edge_index] is False:

                # Variable for temporary Edge disabling and then enable the edge
                # in question (for the example above: edge ID 7 or 8)
                temp_edge_enabled = self.edge_enabled.copy()
                temp_edge_enabled[current_edge_index] = True

                # List for storing all enabled edges,
                # to use for finding out if the graph is fully connected
                edge_vertex_id_pairs_enabled = []

                # For loop for finding all enabled edges
                for edge_to_check in self.edge_vertex_id_pairs:
                    edge_to_check_index = self.edge_vertex_id_pairs.index(edge_to_check)

                    # Leave out the input edge, since that one will be disabled
                    if temp_edge_enabled[edge_to_check_index] and edge_to_check_index != edge_index:
                        edge_vertex_id_pairs_enabled.append(edge_to_check)

                # Check for fully connected-ness
                networkx_graph = nx.Graph()
                networkx_graph.add_nodes_from(self.vertex_ids)
                networkx_graph.add_edges_from(edge_vertex_id_pairs_enabled)
                if nx.is_connected(networkx_graph):
                    try:
                        nx.find_cycle(networkx_graph)
                    except nx.NetworkXNoCycle:
                        output.append(edge)

        return output
