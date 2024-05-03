"""
Graph processing file, handling the low level graph calculations
for finding alternative edges and downstream vertices
"""

from typing import List, Tuple

# import matplotlib.pyplot as plt
import networkx as nx


class IDNotFoundError(Exception):
    """
    Class for the ID not found error, is thrown when a vertex or edge doesnt exist
    """

    def __init__(self, type):
        Exception.__init__(
            self,
            """IDNotFoundError: vertex pair does not exist in
                           vertex_ids (if 0) or source_vertex_id does not exist in
                           vertex_ids (if 1): T"""
            + str(type),
        )


class InputLengthDoesNotMatchError(Exception):
    """
    Class for the InputLengthDoesNotMatchError error, is thrown
    when the input edge or vertex ids and the pairs do not match
    """

    def __init__(self, type, l_edge_vertex_id_pairs, l_edge_ids):
        Exception.__init__(
            self,
            """InputLengthDoesNotMatchError: The length of edge_enabled
                        (if 0) or edge_vertex_id_pairs (if 1) ("""
            + str(l_edge_vertex_id_pairs)
            + ") is different than edge_ids ("
            + str(l_edge_ids)
            + "): T"
            + str(type),
        )


class IDNotUniqueError(Exception):
    """
    Class for the IDNotUniqueError error, is thrown
    when some ID(s) occur more than once in the input variables
    """

    def __init__(self, type):
        Exception.__init__(
            self,
            """IDNotUniqueError: There are non-unique vertices
                           (if 0) or edges (if 1) in the graph: T"""
            + str(type),
        )


class GraphNotFullyConnectedError(Exception):
    """
    Class for the GraphNotFullyConnectedError error, is thrown
    when the graph has 'islands' of vertices, ones that are disconnedted from the main graph.
    """

    def __init__(self):
        Exception.__init__(self, "The graph is not fully connected.")


class GraphCycleError(Exception):
    """
    Class for the GraphCycleError error, is thrown
    when the graph is cyclic.
    """

    def __init__(self):
        Exception.__init__(self, "The graph contains cycles.")


class EdgeAlreadyDisabledError(Exception):
    """
    Class for the EdgeAlreadyDisabledError error, is thrown when find_alternative_edge
    tries to find an alternative edge that is already disabled.
    """

    def __init__(self, edge_disabled_id):
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
        self.vertex_ids = vertex_ids
        self.edge_ids = edge_ids
        self.edge_vertex_id_pairs = edge_vertex_id_pairs
        self.edge_enabled = edge_enabled
        self.source_vertex_id = source_vertex_id

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

    def find_alternative_edges(self, disabled_edge_id: int) -> List[int]:
        """
        Given an enabled edge, do the following analysis:
            If the edge is going to be disabled,
                which (currently disabled) edge can be enabled to ensure
                that the graph is again fully connected and acyclic?
            Return a list of all alternative edges.
        If the disabled_edge_id is not a valid edge id, it should raise IDNotFoundError.
        If the disabled_edge_id is already disabled, it should raise EdgeAlreadyDisabledError.
        If there are no alternative to make the graph
        fully connected again, it should return empty list.

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
        # QUESTIONS
        # What to give as an ouput if you need to enable multiple edges to
        # be enabled to make it work for example: 7 AND 8 need to
        # be enabled to work instead of 7 OR 8
        # Is a specific output order required?

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
