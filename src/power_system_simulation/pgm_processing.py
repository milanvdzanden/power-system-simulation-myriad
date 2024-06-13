"""
Implementation of a power grid simulation and calculation module using the
power-grid-model package as the core. Additional functions are included to display the data. 
"""

from typing import Union

import numpy as np
import pandas as pd
import power_grid_model as pgm

# from power_grid_model.validation import *
from power_grid_model import CalculationMethod
from power_grid_model.utils import json_deserialize, json_serialize
from power_grid_model.validation import ValidationException
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
            in either profiles do not match the node IDs in the PGM JSON descriptor (if 2)"""
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

        pgm.validation.assert_valid_input_data(self.pgm_input)

        self.pgm_model = pgm.PowerGridModel(self.pgm_input)

        # Initialize variables to be used later
        self.update_index_length = 0
        self.update_ids = 0
        self.update_load_profile = 0
        self.time_series_mutation = 0
        self.output_data = 0

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
        pgm.validation.assert_valid_batch_data(
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
