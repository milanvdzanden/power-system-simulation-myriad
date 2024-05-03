"""
This is a skeleton for the graph processing assignment.

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

    def __init__(
        self,
        vertex_ids: List[int],
        edge_ids: List[int],
        edge_vertex_id_pairs: List[Tuple[int, int]],
        edge_enabled: List[bool],
        source_vertex_id: int,
    ) -> None:
        # put your implementation here
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

        for c in enumerate(nx.connected_components(graph)):
            list_of_islands_before.append(list(c))

        # Step 2 is calculated here
        # For loop for finding all enabled edges without the input edge
        edge_vertex_id_pairs_enabled_after = []
        edge_ids_enabled_after = []  # The list with the input edge removed
        for edge_index in enumerate(edge_ids_enabled_before):
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

        for c in enumerate(nx.connected_components(graph)):
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
