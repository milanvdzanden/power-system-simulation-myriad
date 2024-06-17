"""
Implementation of a power grid simulation and calculation module using the
power-grid-model package as the core. Additional functions are included to display the data. 
"""

from typing import Union

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import power_grid_model as pgm

from power_grid_model.validation import assert_valid_input_data, assert_valid_batch_data
from power_grid_model import CalculationMethod
from scipy import integrate

# ValidationError: included in package


class ProfilesDontMatchError(Exception):
    """
    Error class for class ProfilesDontMatchError
    """

    def __init__(self, mode):
        """
        Prints the exception message using the standard exception class.

        Args:
          self: passes exception
          mode: error type,
            0 if the time stamps (time index) do not match
            1 if the node ids do not match each other in the profiles
            2 if the node ids in either profiles do not match the node ids in the pgm descriptor
        """
        Exception.__init__(
            self,
            """ProfilesDontMatchError: The time stamps of the profiles do not match (if 0) or 
            the node IDs do not match eatch other in the profiles (if 1) or the the node IDs 
            in either profiles do not match the node IDs in the PGM JSON descriptor (if 2): T"""
            + str(mode),
        )


class PgmProcessor:
    """
    Class that implements generic handling functionality for Power Grid Model (PGM) data structures for purposes of power flow analysis.
    The primary purpose is to construct a Power Grid Model to an appropriate form, verify the validity of data, and run a batch time-series power flow calculation, obtaining two aggregate tables with information of interest.
    To ease the understanding of the results, a method is available to graph them using NetworkX.
    """

    def __init__(
        self,
        network_data: dict[str, Union[np.ndarray, dict[str, np.ndarray]]],
        active_profile: pd.DataFrame,
        reactive_profile: pd.DataFrame,
    ):
        """
        Initialization method for the PGM processor. The data is assumed to be loaded from .parquet files elsewhere.
        The data is validated using PGM base validation functions to create a Power Grid Model. Miscellaneous variables are also initialized for later use.
        Args:
            network_data: deserialized .json network data in PGM input format
            active_profile: Pandas DataFrame, describing the time-series active load profile mutation for batch power flow calculations
            reactive: Pandas DataFrame, describing the time-series reactive load profile mutation for batch power flow calculations
        Raises:
            ValidationExceptionError: if input data in invalid
        """
        self.pgm_input = network_data
        self.active_load_profile = active_profile
        self.reactive_load_profile = reactive_profile

        assert_valid_input_data(self.pgm_input)

        self.pgm_model = pgm.PowerGridModel(self.pgm_input)

        # Initialize variables to be used later
        self.update_index_length = 0
        self.update_ids = 0
        self.update_load_profile = 0
        self.time_series_mutation = 0
        self.output_data = 0

        # Indicate that the graph for drawing aggregate and output data is not ready
        self.draw_ready = False
        self.output_ready = False

        # Define drawing self-attributes
        # These variables are internal as animation handling using matplotlib does not work properly
        # With functions that pass any attributes other than frame
        self.draw_g = None
        self.draw_aggregate_table = None
        self.draw_ax = None

        self.cmap_reds = None
        self.cmap_blues = None

        self.draw_pos = None

        self.draw_node_size = 0
        self.draw_node_connecting_size = 0
        self.draw_line_width = 0.0
        self.draw_line_disabled_width = 0.0

        self.draw_pf_max_line = 0
        self.draw_pf_min_line = 0
        self.draw_pf_max_load = 0
        self.draw_pf_min_load = 0

        self.draw_bus = None

        self.draw_pf_line_bb = None
        self.draw_pf_line_bl = None


    def create_update_model(
        self, use_active_load_profile: bool = None, use_reactive_load_profile: bool = None
    ) -> None:
        """
        Validates update data and creates the time-series batch mutations of active and reactive load profiles of network nodes for power-flow analysis.
        Args:
            use_active_load_profile: if specificed, a different active load profile can be used than the one the class was initialized with.
            use_reactive_load_profile: if specificed, a different active load profile can be used than the one the class was initialized with.
        Raises:
            ProfilesDontMatchError(0): if time series of both active and reactive profiles don't match
            ProfilesDontMatchError(1): if node IDs are not the same in either profile
            ProfilesDontMatchError(2): if node IDs are not the same in either profile, and in the Power Grid Model .json network description
        """
        use_active_load_profile = use_active_load_profile or self.active_load_profile
        use_reactive_load_profile = use_reactive_load_profile or self.reactive_load_profile

        # Check if time series of both active and reactive profile match
        if not use_active_load_profile.index.equals(use_reactive_load_profile.index):
            raise ProfilesDontMatchError(0)

        # Check if node IDs match in both profiles
        if not use_active_load_profile.columns.equals(use_reactive_load_profile.columns):
            raise ProfilesDontMatchError(1)

        # Check if node IDs in both profiles match the node IDs in the PGM JSON input descriptor
        if not np.array_equal(
            pd.DataFrame(self.pgm_input["sym_load"]).loc[:, "id"].to_numpy(),
            use_active_load_profile.columns.to_numpy(),
        ):
            raise ProfilesDontMatchError(2)

        # Validated, take any
        self.update_index_length = use_active_load_profile.index.shape[0]
        self.update_ids = use_active_load_profile.columns.to_numpy()

        self.update_load_profile = pgm.initialize_array(
            "update", "sym_load", (self.update_index_length, self.update_ids.shape[0])
        )
        self.update_load_profile["id"] = self.update_ids
        self.update_load_profile["p_specified"] = use_active_load_profile
        self.update_load_profile["q_specified"] = use_reactive_load_profile
        self.update_load_profile["status"] = 1

        self.time_series_mutation = {"sym_load": self.update_load_profile}
        assert_valid_batch_data(
            input_data=self.pgm_input,
            update_data=self.time_series_mutation,
            calculation_type=pgm.CalculationType.power_flow,
        )

    def run_batch_process(self) -> None:
        """
        Run the batch process on the input data according to the load update profile. Results are stored in a class variable.
        """
        self.output_data = self.pgm_model.calculate_power_flow(
            update_data=self.time_series_mutation,
            calculation_method=CalculationMethod.newton_raphson,
        )

    def get_aggregate_results(self) -> list[pd.DataFrame, pd.DataFrame]:
        """
        Generate the two required output tables based on the variables created in run_batch_process.
        Returns:
            2-element list of data frames for the 2 required aggregated tables
        """
        # Initialize output variable
        aggregate_results = [None, None]

        # Make a list of all timestamps for later use
        list_of_timestamps = self.active_load_profile.index.strftime("%Y-%m-%d %H:%M:%S").to_list()

        # Output dataframe for the first required table
        df_min_max_nodes = pd.DataFrame()

        # Loop through all timestamps, and pick the nodes with minimum and maximum voltage
        for index, snapshot in enumerate(self.output_data["node"]):

            # Temporary dataframe which contains the timestamp snapshot data
            df = pd.DataFrame(snapshot)

            # Find the index of the row with the minimum and maximum value in the 'u_pu' column
            max_index = df["u_pu"].idxmax()
            min_index = df["u_pu"].idxmin()

            # Retrieve the row with the minimum and maximum value in the 'u_pu' column
            max_row = df.loc[max_index]
            min_row = df.loc[min_index]

            # Put the data in the correct rows, and columns
            df_min_max_nodes.loc[list_of_timestamps[index], "Max_Voltage_Node"] = max_row["id"]
            df_min_max_nodes.loc[list_of_timestamps[index], "Max_Voltage"] = max_row["u_pu"]
            df_min_max_nodes.loc[list_of_timestamps[index], "Min_Voltage_Node"] = min_row["id"]
            df_min_max_nodes.loc[list_of_timestamps[index], "Min_Voltage"] = min_row["u_pu"]

        aggregate_results[0] = df_min_max_nodes

        # Make a list with all the timestamp snapshots, to be used to find the max and min
        flattened_list = [tuple for sublist in self.output_data["line"] for tuple in sublist]
        data = [
            {"id": tpl[0], "energized": tpl[1], "loading": tpl[2], "p_from": tpl[3], "p_to": tpl[7]}
            for tpl in flattened_list
        ]

        # Output dataframe for the second required output table
        df_line_loss = pd.DataFrame()

        df = pd.DataFrame(data)

        # Number of unique lines
        num_nodes = df["id"].nunique()
        repeated_list_of_timestamps = [
            elem for elem in list_of_timestamps for _ in range(num_nodes)
        ]

        # Add timestamp column to dataframe
        df["Timestamp"] = repeated_list_of_timestamps

        # Group data by each line and then loop over each dataframe 'group'
        grouped_by_line = df.groupby("id")

        for line_id, line in grouped_by_line:
            line["p_loss"] = abs(abs(line["p_from"]) - abs(line["p_to"]))

            # Calculate the area under the power loss curve
            energy_loss = integrate.trapezoid(line["p_loss"].to_list()) / 1000

            # Find the index of the row with the minimum and maximum value in the 'u_pu' column
            max_index = line["loading"].idxmax()
            min_index = line["loading"].idxmin()

            # Retrieve the row with the minimum and maximum value in the 'u_pu' column
            max_row = line.loc[max_index]
            min_row = line.loc[min_index]

            # Put the data in the correct rows, and columns
            df_line_loss.loc[line_id, "Total_Loss"] = energy_loss
            df_line_loss.loc[line_id, "Max_Loading_Timestamp"] = max_row["Timestamp"]
            df_line_loss.loc[line_id, "Max_Loading"] = max_row["loading"]
            df_line_loss.loc[line_id, "Min_Loading_Timestamp"] = min_row["Timestamp"]
            df_line_loss.loc[line_id, "Min_Loading"] = min_row["loading"]

        aggregate_results[1] = df_line_loss
        aggregate_results[1].index.name = "Line_ID"
        aggregate_results[0].index = pd.to_datetime(aggregate_results[0].index)
        aggregate_results[0].index.name = "Timestamp"

        return aggregate_results

    def compare_to_expected(
        self,
        aggregate_results: list[pd.DataFrame, pd.DataFrame],
        expected_per_line: pd.DataFrame,
        expected_per_timestamp: pd.DataFrame,
    ) -> bool:
        """
        Compare the aggregated output tables to other aggeregated output tables of the same format.
        This method can be used to compare to pre-calculated values for checking algorithm validity, or for comparsion with hand-calculations, PowerWorld simulations, SimuLink simulations, etc.
        Args:
            aggregate_results: 2-element list of data frames for the 2 required aggregated tables
            expected_per_line: Expected output for the per-line quantities (loading and losses)
            expected_per_timestamp: Expected output for the per-timestamp quantities (voltage records)
        Returns:
            True if both aggregate tables are equal, false otherwise.
        """
        aggregate_results[1]["Max_Loading_Timestamp"] = pd.Series(
            pd.to_datetime(aggregate_results[1]["Max_Loading_Timestamp"])
        )
        aggregate_results[1]["Min_Loading_Timestamp"] = pd.Series(
            pd.to_datetime(aggregate_results[1]["Min_Loading_Timestamp"])
        )
        try:
            pd.testing.assert_frame_equal(
                aggregate_results[1],
                expected_per_line,
                check_dtype=False,
                check_index_type=False,
                check_exact=False,
                check_like=True,
                atol=1e-8,
            )
            pd.testing.assert_frame_equal(
                aggregate_results[0],
                expected_per_timestamp,
                check_dtype=False,
                check_index_type=False,
                check_exact=False,
                check_like=True,
                atol=1e-8,
            )
            return True
        except:
            return False
        return False

    def draw_to_networkx(
        self, aggregate_results: list[pd.DataFrame, pd.DataFrame], draw_ax: matplotlib
    ) -> None:
        """
        Creates a NetworkX graph specifically for drawing, assigning attributes
        to nodes (bues) and edges (lines) to differentiate them in drawing.

        Once this method is called, the class object is ready for drawing.
        Args:
            aggregate_results: aggregate results from power flow analysis; obtained with
                other functions
            draw_ax: matplotlib axes object for drawing, used to initialize the internal
                class variable
        """
        # Initialize internal variables
        self.draw_g = nx.Graph()
        self.draw_aggregate_table = aggregate_results
        self.draw_ax = draw_ax

        if "node" in self.pgm_input:
            # Add all nodes, based on "id".
            for node in self.pgm_input["node"]:
                self.draw_g.add_node(node[0], type="default")

        if "sym_load" in self.pgm_input and "source" in self.pgm_input:
            # Classify nodes based on source or load
            for sym_load in self.pgm_input["sym_load"]:
                self.draw_g.nodes[sym_load[1]].update({"type": "sym_load"})
            for source in self.pgm_input["source"]:
                self.draw_g.nodes[source[1]].update({"type": "source"})

        if "line" in self.pgm_input:
            # Add all edges/lines, based on "from_node", "to_node".
            for line in self.pgm_input["line"]:
                # Bus-Bus or Bus-Load node
                if nx.get_node_attributes(self.draw_g, "type")[line[2]] == "sym_load":
                    self.draw_g.add_edge(line[1], line[2], id=line[0], type="to_load")
                else:
                    self.draw_g.add_edge(line[1], line[2], id=line[0], type="bus_bus")
                # Enabled or disabled edge
                if line[3] == 0 or line[4] == 0:
                    self.draw_g.edges[line[1], line[2]].update({"enabled": False})
                else:
                    self.draw_g.edges[line[1], line[2]].update({"enabled": True})

        if "transformer" in self.pgm_input:
            # Add transformer lines (these are separate in the data from the lines)
            for transformer in self.pgm_input["transformer"]:
                self.draw_g.add_edge(
                    transformer[1], transformer[2], id=transformer[0], type="transformer"
                )

        # Create colormaps (since <0.1 tend to disappear in exported gif, most likely due to compression)
        self.cmap_reds = matplotlib.colors.LinearSegmentedColormap.from_list(
            "cmap_reds", matplotlib.pyplot.cm.Reds(np.linspace(0.2, 1.0))
        )
        self.cmap_blues = matplotlib.colors.LinearSegmentedColormap.from_list(
            "cmap_blues", matplotlib.pyplot.cm.Blues(np.linspace(0.2, 1.0))
        )

        # Optional validity checks
        self.draw_ready = True

    def draw_init(self) -> None:
        """
        Internal method to avoid redunant repeating
        creations of drawing layouts and variables.

        Kamada Kawai layout has been found to be the most interesting-looking layout
        Planar and BFS layouts are also good in their own way, but struggle with
        Accurately displaying large networks
        """
        if self.draw_ready:
            self.draw_pos = nx.kamada_kawai_layout(self.draw_g)

            self.draw_node_size = 100
            self.draw_node_connecting_size = 300
            self.draw_line_width = 2.1
            self.draw_line_disabled_width = 0.7

        else:
            raise SystemError("Drawing network is uninitialized")

    def draw_init_power_flow(self, draw_bus: str = "u_pu") -> None:
        """
        Initializes the power flow to avoid re-calculating the color scheme for the animation
        every frame.
        Args:
            draw_bus: chosen quantity to color-map to on the animation for buses
        """
        if not self.draw_ready:
            raise SystemError("Drawing network is uninitialized or output is not calculated")
        # Initialize variables
        self.draw_pf_max_line = 0
        self.draw_pf_min_line = 1e33
        self.draw_pf_max_load = 0
        self.draw_pf_min_load = 1e33

        self.draw_bus = draw_bus

        # Get maximum line loading over time
        for x in self.output_data["line"]:
            if 100 * max(x["loading"] > self.draw_pf_max_line):
                self.draw_pf_max_line = 100 * max(x["loading"])
        # Get minimum line loading over time
        for x in self.output_data["line"]:
            if 100 * min(x["loading"] < self.draw_pf_min_line):
                self.draw_pf_min_line = 100 * min(x["loading"])
        # Get maximum sym load quantity over time
        for x in self.output_data["node"]:
            if max(x[draw_bus] > self.draw_pf_max_load):
                self.draw_pf_max_load = max(x[draw_bus])
        # Get minimum sym load quantity over time
        for x in self.output_data["node"]:
            if min(x[draw_bus] < self.draw_pf_min_load):
                self.draw_pf_min_load = min(x[draw_bus])

        # Get list of bus-bus lines. Assumed that these are valid lines, in same order as self.draw_g.edges()
        draw_pf_line_bb_mask = [
            key
            for key, value in nx.get_edge_attributes(self.draw_g, "type").items()
            if value == "bus_bus"
        ]
        self.draw_pf_line_bb = {
            key: value
            for key, value in nx.get_edge_attributes(self.draw_g, "id").items()
            if key in draw_pf_line_bb_mask
        }
        # Get list of bus-load lines. Assumed that these are valid lines, in same order as self.draw_g.edges()
        draw_pf_line_bl_mask = [
            key
            for key, value in nx.get_edge_attributes(self.draw_g, "type").items()
            if value == "to_load"
        ]
        self.draw_pf_line_bl = {
            key: value
            for key, value in nx.get_edge_attributes(self.draw_g, "id").items()
            if key in draw_pf_line_bl_mask
        }
        # Draw bus draw on-top of every node (source, bus, sym_load)

        # Add colormaps
        sm = matplotlib.pyplot.cm.ScalarMappable(
            cmap=self.cmap_reds,
            norm=matplotlib.pyplot.Normalize(
                vmin=self.draw_pf_min_line, vmax=self.draw_pf_max_line
            ),
        )
        matplotlib.pyplot.colorbar(
            sm, ax=self.draw_ax, shrink=0.75, label="Line loading [%]", anchor=(0, 1)
        )
        sm = matplotlib.pyplot.cm.ScalarMappable(
            cmap=self.cmap_blues,
            norm=matplotlib.pyplot.Normalize(
                vmin=self.draw_pf_min_load, vmax=self.draw_pf_max_load
            ),
        )
        matplotlib.pyplot.colorbar(
            sm, ax=self.draw_ax, shrink=0.75, label="Node voltage [p.u.]", anchor=(0.2, 1)
        )

        self.output_ready = True

    def __draw_basic(self):
        """
        Internal method that draws an uncolored, single-frame for a network.
        Used repeatedly within animations.
        """
        if not self.draw_ready:
            raise SystemError("Drawing network is uninitialized")
        # --- --- ---

        # Draw generic (connection) nodes
        draw_node_list = [
            key
            for key, value in nx.get_node_attributes(self.draw_g, "type").items()
            if value == "default"
        ]
        nx.draw_networkx_nodes(
            self.draw_g,
            ax=self.draw_ax,
            nodelist=draw_node_list,
            pos=self.draw_pos,
            node_size=self.draw_node_connecting_size,
            node_shape="|",
            label="Buses",
            node_color="k",
        )
        # Draw load nodes
        draw_node_list = [
            key
            for key, value in nx.get_node_attributes(self.draw_g, "type").items()
            if value == "sym_load"
        ]
        nx.draw_networkx_nodes(
            self.draw_g, ax=self.draw_ax, nodelist=draw_node_list, pos=self.draw_pos, node_size=0
        )
        # Draw source nodes
        draw_node_list = [
            key
            for key, value in nx.get_node_attributes(self.draw_g, "type").items()
            if value == "source"
        ]
        nx.draw_networkx_nodes(
            self.draw_g,
            ax=self.draw_ax,
            nodelist=draw_node_list,
            pos=self.draw_pos,
            node_size=self.draw_node_size,
            node_shape="o",
            label="Sources",
            node_color="k",
        )

        # Draw bus-load lines (assume all enabled)
        draw_edge_list_bl = [
            key
            for key, value in nx.get_edge_attributes(self.draw_g, "type").items()
            if value == "to_load"
        ]
        nx.draw_networkx_edges(
            self.draw_g,
            ax=self.draw_ax,
            edgelist=draw_edge_list_bl,
            pos=self.draw_pos,
            node_size=self.draw_node_connecting_size,
            width=self.draw_line_width,
            label="Symmetrical Loads",
            arrows=True,
            arrowstyle="->",
            edge_color="brown",
        )

        draw_edge_enabled = [
            key
            for key, value in nx.get_edge_attributes(self.draw_g, "enabled").items()
            if value == True
        ]
        # Draw enabled bus-bus lines
        draw_edge_list_bb = [
            key
            for key, value in nx.get_edge_attributes(self.draw_g, "type").items()
            if value == "bus_bus"
            if key in draw_edge_enabled
        ]
        nx.draw_networkx_edges(
            self.draw_g,
            ax=self.draw_ax,
            edgelist=draw_edge_list_bb,
            pos=self.draw_pos,
            node_size=self.draw_node_connecting_size,
            width=self.draw_line_width,
            label="Lines",
            connectionstyle="angle,angleA=-90,angleB=180",
            arrows=True,
        )
        # Draw disabled bus-bus lines
        draw_edge_list_bb = [
            key
            for key, value in nx.get_edge_attributes(self.draw_g, "type").items()
            if value == "bus_bus"
            if key not in draw_edge_enabled
        ]
        nx.draw_networkx_edges(
            self.draw_g,
            ax=self.draw_ax,
            edgelist=draw_edge_list_bb,
            pos=self.draw_pos,
            node_size=self.draw_node_connecting_size,
            width=self.draw_line_disabled_width,
            edge_color="grey",
            label="Lines (disabled)",
        )
        # Draw transformer lines
        draw_edge_list_t = [
            key
            for key, value in nx.get_edge_attributes(self.draw_g, "type").items()
            if value == "transformer"
        ]
        nx.draw_networkx_edges(
            self.draw_g,
            ax=self.draw_ax,
            edgelist=draw_edge_list_t,
            pos=self.draw_pos,
            node_size=self.draw_node_size,
            style="dashed",
            width=self.draw_line_width,
            label="Transformer Lines",
        )

        # Draw load labels
        nx.draw_networkx_labels(
            self.draw_g,
            ax=self.draw_ax,
            pos=self.draw_pos,
            verticalalignment="top",
            font_size=8,
            font_color="brown",
            labels={
                key: key
                for key, value in nx.get_node_attributes(self.draw_g, "type").items()
                if value == "sym_load"
            },
        )
        # Draw bus labels
        nx.draw_networkx_labels(
            self.draw_g,
            ax=self.draw_ax,
            pos=self.draw_pos,
            horizontalalignment="left",
            font_size=8,
            font_color="black",
            labels={
                key: key
                for key, value in nx.get_node_attributes(self.draw_g, "type").items()
                if value == "default"
            },
        )

    def __draw_timelapse_power_flow(self, frame: int) -> None:
        """
        Internal method that draw power-flow related elements.
        Args:
            frame: integer, handled by matplotlib.animation.FuncAnimation
        """
        # Initialize variables
        line_list_colors_value_bb = {}
        line_list_colors_value_bl = {}
        node_list_colors_value = []

        # Create and paint over bus-bus connections
        for index, row in pd.DataFrame(self.output_data["line"][frame]).iterrows():
            for key, value in self.draw_pf_line_bb.items():
                if row["id"] == value:
                    line_list_colors_value_bb[key] = 100 * row["loading"]
        nx.draw_networkx_edges(
            self.draw_g,
            ax=self.draw_ax,
            edgelist=line_list_colors_value_bb.keys(),
            pos=self.draw_pos,
            node_size=self.draw_node_size,
            edge_color=line_list_colors_value_bb.values(),
            edge_cmap=self.cmap_reds,
            connectionstyle="angle,angleA=-90,angleB=180",
            arrows=True,
            edge_vmin=self.draw_pf_min_line,
            edge_vmax=self.draw_pf_max_line,
            width=3,
        )
        # Create and paint over bus-load connections
        for comb in pd.DataFrame(self.output_data["line"][frame]).iterrows():
            for key, value in self.draw_pf_line_bl.items():
                if comb[1]["id"] == value:
                    line_list_colors_value_bl[key] = 100 * comb[1]["loading"]
        nx.draw_networkx_edges(
            self.draw_g,
            ax=self.draw_ax,
            edgelist=line_list_colors_value_bl.keys(),
            pos=self.draw_pos,
            node_size=self.draw_node_size,
            edge_color=line_list_colors_value_bl.values(),
            edge_cmap=self.cmap_reds,
            arrows=True,
            arrowstyle="->",
            edge_vmin=self.draw_pf_min_line,
            edge_vmax=self.draw_pf_max_line,
            width=3.5,
        )
        # Create and paint over nodes (buses, loads, sources)
        for index, row in pd.DataFrame(self.output_data["node"][frame]).iterrows():
            node_list_colors_value.append(row[self.draw_bus])
        nx.draw_networkx_nodes(
            self.draw_g,
            ax=self.draw_ax,
            pos=self.draw_pos,
            node_size=0.25 * self.draw_node_size,
            node_shape="s",
            node_color=node_list_colors_value,
            cmap=matplotlib.colormaps.get_cmap("Blues"),
            vmin=self.draw_pf_min_load,
            vmax=self.draw_pf_max_load,
        )

    def __draw_timelapse_node_loading(self, frame: int) -> None:
        """
        Internal method that draw node-loading related elements.
        Args:
            frame: integer, handled by matplotlib.animation.FuncAnimation
        """
        # Get maximum voltage and where
        node_maxv = [int(self.draw_aggregate_table[0].iloc[frame, :]["Max_Voltage_Node"])]
        maxv = "%.3f" % round(self.draw_aggregate_table[0].iloc[frame, :]["Max_Voltage"], 3)
        # Get minimum voltage and where
        node_minv = [int(self.draw_aggregate_table[0].iloc[frame, :]["Min_Voltage_Node"])]
        minv = "%.3f" % round(self.draw_aggregate_table[0].iloc[frame, :]["Min_Voltage"], 3)
        # Draw maximum/minimum voltage
        maxv_label = "> Max Voltage : " + maxv
        nx.draw_networkx_nodes(
            self.draw_g,
            ax=self.draw_ax,
            nodelist=node_maxv,
            pos=self.draw_pos,
            node_size=self.draw_node_size,
            node_shape="s",
            label=maxv_label,
            node_color="red",
        )
        minv_label = "> Min Voltage : " + minv
        nx.draw_networkx_nodes(
            self.draw_g,
            ax=self.draw_ax,
            nodelist=node_minv,
            pos=self.draw_pos,
            node_size=self.draw_node_size,
            node_shape="s",
            label=minv_label,
            node_color="green",
        )

    def draw_timelapse_power_flow(self, frame: int) -> None:
        """
        Uses MatPlotLib to draw an animation for the power flow.
        Args:
            frame: integer, handled by matplotlib.animation.FuncAnimation
        """
        if not self.draw_ready or not self.output_ready:
            raise SystemError(
                "Drawing network is uninitialized or output is not calculated. Call draw_init_power_flow before animation"
            )

        self.draw_ax.clear()
        self.__draw_basic()
        self.__draw_timelapse_power_flow(frame - 1)

        # Draw legend
        self.draw_ax.legend(loc=(1.05, 0))

        # Add title
        self.draw_ax.set_title(
            "Time: " + str(self.draw_aggregate_table[0].index[frame - 1]) + " [" + str(frame) + "]"
        )

    def draw_timelapse_node_loading(self, frame: int) -> None:
        """
        Uses MatPlotLib to draw an animation for the node (bus) voltage loading
        Args:
            frame: integer, handled by matplotlib.animation.FuncAnimation
        """
        if not self.draw_ready:
            raise SystemError("Drawing network is uninitialized")
        self.draw_ax.clear()
        self.__draw_basic()
        self.__draw_timelapse_node_loading(frame - 1)

        # Draw legend
        self.draw_ax.legend(loc=(1.05, 0))

        # Add title
        self.draw_ax.set_title(
            "Time: " + str(self.draw_aggregate_table[0].index[frame - 1]) + " [" + str(frame) + "]"
        )

    def draw_static_line_loading(self, criterion: str) -> None:
        """
        Uses MatPlotLib to draw a static frame representation of the line loading, colored by loading.
        Args:
            criterion: Show total loss, maximum loading, or minimum loading
            Valid arguments: 'power_loss', 'max_loading', 'min_loading'.
        """
        if not self.draw_ready:
            raise SystemError("Drawing network is uninitialized")
        self.draw_ax.clear()
        self.draw_init()
        self.__draw_basic()

        # Get list of all generic lines (to color)
        line_list_keys = {
            key
            for key, value in nx.get_edge_attributes(self.draw_g, "type").items()
            if value != "transformer"
        }
        line_list = {
            key: value
            for key, value in nx.get_edge_attributes(self.draw_g, "id").items()
            if key in line_list_keys
        }
        line_list_colors = {}

        match criterion:
            case "power_loss":
                for edge_id, row in self.draw_aggregate_table[1].iterrows():
                    for key, value in line_list.items():
                        if value == edge_id:
                            line_list_colors[key] = row["Total_Loss"]
                self.draw_ax.set_title("Total power loss per line for the network, whole timeline")
                bar_title = "Total Power Loss [kWh]"
            case "max_loading":
                for edge_id, row in self.draw_aggregate_table[1].iterrows():
                    for key, value in line_list.items():
                        if value == edge_id:
                            line_list_colors[key] = row["Max_Loading"]
                self.draw_ax.set_title("Maximum loading per line for the network, whole timeline")
                bar_title = "Maximum Loading [p.u]"
            case "min_loading":
                for edge_id, row in self.draw_aggregate_table[1].iterrows():
                    for key, value in line_list.items():
                        if value == edge_id:
                            line_list_colors[key] = row["Min_Loading"]
                self.draw_ax.set_title("Minimum loading per line for the network, whole timeline")
                bar_title = "Minimum Loading [p.u]"
            case _:
                raise ValueError("Invalid static line drawing representation type provided.")

        # Set color boundaries for colormap
        colors_value_max = max(line_list_colors.values())
        colors_value_min = min(line_list_colors.values())

        # Create and paint over bus-bus connections
        line_list_colors_key_bb = [
            key
            for key, value in line_list_colors.items()
            if key
            in [
                key
                for key, value in nx.get_edge_attributes(self.draw_g, "type").items()
                if value == "bus_bus"
            ]
        ]
        line_list_colors_value_bb = [
            value
            for key, value in line_list_colors.items()
            if key
            in [
                key
                for key, value in nx.get_edge_attributes(self.draw_g, "type").items()
                if value == "bus_bus"
            ]
        ]
        nx.draw_networkx_edges(
            self.draw_g,
            ax=self.draw_ax,
            edgelist=line_list_colors_key_bb,
            pos=self.draw_pos,
            node_size=self.draw_node_size,
            edge_color=line_list_colors_value_bb,
            edge_cmap=self.cmap_reds,
            connectionstyle="angle,angleA=-90,angleB=180",
            arrows=True,
            edge_vmin=colors_value_min,
            edge_vmax=colors_value_max,
            width=3,
        )
        # Create and paint over bus-load connections
        line_list_colors_key_bl = [
            key
            for key, value in line_list_colors.items()
            if key
            in [
                key
                for key, value in nx.get_edge_attributes(self.draw_g, "type").items()
                if value == "to_load"
            ]
        ]
        line_list_colors_value_bl = [
            value
            for key, value in line_list_colors.items()
            if key
            in [
                key
                for key, value in nx.get_edge_attributes(self.draw_g, "type").items()
                if value == "to_load"
            ]
        ]
        nx.draw_networkx_edges(
            self.draw_g,
            ax=self.draw_ax,
            edgelist=line_list_colors_key_bl,
            pos=self.draw_pos,
            node_size=self.draw_node_size,
            edge_color=line_list_colors_value_bl,
            edge_cmap=self.cmap_reds,
            arrows=True,
            arrowstyle="->",
            edge_vmin=colors_value_min,
            edge_vmax=colors_value_max,
            width=3.5,
        )

        # Draw maximum/minimum record (Scuffy) - don't round, because the numbers are very small
        max_label = "> Max : " + str(colors_value_max)
        matplotlib.pyplot.plot([], [], " ", label=max_label)
        min_label = "> Min : " + str(colors_value_min)
        matplotlib.pyplot.plot([], [], " ", label=min_label)
        sm = matplotlib.pyplot.cm.ScalarMappable(
            cmap=self.cmap_reds,
            norm=matplotlib.pyplot.Normalize(vmin=colors_value_min, vmax=colors_value_max),
        )
        matplotlib.pyplot.colorbar(sm, ax=self.draw_ax, shrink=0.5, label=bar_title)

        # Draw legend
        self.draw_ax.legend(loc=(1.05, 0))

    def draw_static_simple_load_active(self, data: dict[str, np.ndarray], load_id: int, ax) -> None:
        """
        Uses MatPlotLib to draw a static, simple line plot of active loading for a specified symmetrical load.
        Args:
            data: output data from a batch power flow analysis
            load_id: load_id to plot
            ax: matplotlib axes object to use
        """
        # Initialize time series
        time_series = []

        # Find id to index
        for index, row in pd.DataFrame(data["sym_load"][0]).iterrows():
            if row["id"] == load_id:
                load_index = index
                break

        # Construct time series
        for x in enumerate(data["sym_load"]):
            time_series.append(pd.DataFrame(x[1]).loc[load_index, "p"] / 1000)

        matplotlib.pyplot.plot(time_series)
        matplotlib.pyplot.xlabel("Batch number [-]")
        matplotlib.pyplot.ylabel("Active power on load [kW]")
        ax.set_title("Active power on symmetrical load " + str(load_id) + " over time")
