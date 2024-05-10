"""
Implementation of a power grid simulation and calculation module using the power-grid-model package as the core. Additional functions are included to display the data. 
"""
import json
import pprint
import warnings
import pyarrow

with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
    # suppress warning about pyarrow as future required dependency
    from pandas import DataFrame
    
import numpy as np
import pandas as pd
import power_grid_model as pgm
from power_grid_model.utils import json_deserialize, json_serialize
from power_grid_model.validation import ValidationException
#from power_grid_model.validation import *
from power_grid_model import *

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
            "ProfilesDontMatchError: The time stamps of the profiles do not match (if 0) or the node IDs do not match eatch other in the profiles (if 1) or the the node IDs in either profiles do not match the node IDs in the PGM JSON descriptor (if 2)" + str(mode),
        )



class PgmProcessor:
    """
    DOCSTRING
    """

    def __init__(self, dir_network_json: str, dir_active_profile: str, dir_reactive_profile: str):
        """
        Write the initialization code to load the 3 files in the argument directories and put them into a power-grid-model model
        You will need to de-serialize the .json input for the input_network_data file (https://power-grid-model.readthedocs.io/en/stable/examples/Serialization%20Example.html)
        Store the data in the self.[...] variables, since then the later functions can access these parameters directly from the class (no need to pass args)
        """
        with open(dir_network_json) as fp:
            data = fp.read()
        dataset = json_deserialize(data)
        self.pgm_input = dataset
        
        # Read active and reactive load profile from parquet file
        self.active_load_profile = pd.read_parquet(dir_active_profile)
        self.reactive_load_profile = pd.read_parquet(dir_reactive_profile)

        """
            Construct the PGM using the input data
            Raises:
            - ValidationException error if input data in invalid
        """
        #try:
        #    self.pgm = pgm.PowerGridModel(self.pgm_input)
        #except pgm.validation.assert_valid_input_data(self.pgm_input):
        #    raise ValidationException
        pgm.validation.assert_valid_input_data(self.pgm_input)

        self.pgm_model = pgm.PowerGridModel(self.pgm_input)

    def create_update_model(self):
        """
        How we think this works is that from the input data (that is read in __init__), you create an update profile that changes the load profiles
        for each timestamp, which allows the batch calculation to run directly.

        Alternatively, it may be also needed that for the batch, you need to "update" on each timestamp. In that case, this function could be made private
        and called between each calculation of the power flow.
        Store results in a self.[...] variable        
        """   
        """
        print(self.pgm_input["node"])  
        print(self.pgm_input["line"]) 
        print(self.pgm_input["sym_load"])   
        print(self.pgm_input["source"])  
        print(self.active_load_profile) 
        print(self.reactive_load_profile)

        # this will likely also not work
        # Check if rows and colums of active and reactive profiles match
        # if self.active_load_profile.shape != self.reactive_load_profile.shape:
        #    raise ProfilesDontMatchError("Number of rows and columns in active and reactive profiles do not match.")
        
        # this will not work
        # Check if load ids and timestamps of active and reactive profiles match
        #if not self.active_load_profile.equals(self.reactive_load_profile):
        #    raise ProfilesDontMatchError("Timestamps and load ids in active and reactive profiles do not match.")
        """
        # Check if time series of both active and reactive profile match
        if not self.active_load_profile.index.equals(self.reactive_load_profile.index):
            raise ProfilesDontMatchError(0)
        
        # Check if node IDs match in both profiles
        if not self.active_load_profile.columns.equals(self.reactive_load_profile.columns):
            raise ProfilesDontMatchError(1)

        # Check if node IDs in both profiles match the node IDs in the PGM JSON input descriptor
        if not np.array_equal(pd.DataFrame(self.pgm_input["sym_load"]).loc[:, "id"].to_numpy(), self.active_load_profile.columns.to_numpy()) or not np.array_equal(pd.DataFrame(self.pgm_input["sym_load"]).loc[:, "id"].to_numpy(), self.reactive_load_profile.columns.to_numpy()):
            raise ProfilesDontMatchError(2)
        
        # Validated, take any
        self.update_index_length = self.active_load_profile.index.shape[0] 
        self.update_ids = self.active_load_profile.columns.to_numpy() 

        self.update_load_profile = pgm.initialize_array("update", "sym_load", (self.update_index_length, self.update_ids.shape[0]))
        self.update_load_profile["id"] = self.update_ids
        self.update_load_profile["p_specified"] = self.active_load_profile
        self.update_load_profile["q_specified"] = self.reactive_load_profile
        #print(self.update_load_profile)

        self.time_series_mutation = {"sym_load": self.update_load_profile}
        pgm.validation.assert_valid_batch_data(input_data=self.pgm_input, update_data=self.time_series_mutation, calculation_type=pgm.CalculationType.power_flow)

    def run_batch_process(self):
        """
        Run the batch process on the input data according to the load update profile.
        Store results in a self.[...] variable
        """        
        self.output_data = self.pgm_model.calculate_power_flow(update_data=self.time_series_mutation)

    def get_aggregate_results(self) -> list[pd.DataFrame, pd.DataFrame]:
        """
        Generate the two required output tables based on the self variables created in run_batch_process
        Output is a 2-element list of data frames for the 2 required aggregated tables
        """        
        # Make a list of all timestamps for later use
        list_of_timestamps = self.active_load_profile.index.strftime('%Y-%m-%d %H:%M:%S').to_list()
        
        # Output dataframe for the first required table 
        df_min_max_nodes = pd.DataFrame()
        
        # Loop through all timestamps, and pick the nodes with minimum and maximum voltage 
        for index, snapshot in enumerate(self.output_data['node']):
            
            # Temporary dataframe which contains the timestamp snapshot data
            df = pd.DataFrame(snapshot)
            
            # Find the index of the row with the minimum and maximum value in the 'u_pu' column
            max_index = df['u_pu'].idxmax()
            min_index = df['u_pu'].idxmin()

            # Retrieve the row with the minimum and maximum value in the 'u_pu' column
            max_row = df.loc[max_index]
            min_row = df.loc[min_index]
            
            # Put the data in the correct rows, and columns
            df_min_max_nodes.loc[list_of_timestamps[index], 'id_max'] = max_row['id']
            df_min_max_nodes.loc[list_of_timestamps[index], 'u_pu_max'] = max_row['u_pu']
            df_min_max_nodes.loc[list_of_timestamps[index], 'id_min'] = min_row['id']
            df_min_max_nodes.loc[list_of_timestamps[index], 'u_pu_min'] = min_row['u_pu']

        print(df_min_max_nodes)
                  
        flattened_list = [tuple for sublist in self.output_data['line'] for tuple in sublist]
        data = [{'id': tpl[0], 'energized': tpl[1], 'loading': tpl[2], 'p_from': tpl[3], 'p_to': tpl[7]} for tpl in flattened_list]
        
        # Output dataframe for the second required output table
        df_line_loss = pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Number of unique lines
        N_nodes = df['id'].nunique()
        repeated_list_of_timestamps = [elem for elem in list_of_timestamps for _ in range(N_nodes)]
        
        # Add timestamp column to dataframe
        df['Timestamp'] = repeated_list_of_timestamps
        
        # Group data by each line and then loop over each dataframe 'group'
        grouped_by_line = df.groupby('id')
        
        for id, line in grouped_by_line:
            print(line)
            
            line['p_loss'] = abs(abs(line['p_from']) - abs(line['p_to']))
            
            # Calculate the area under the power loss curve
            energy_loss = integrate.trapezoid(line['p_loss'].to_list()) / 1000
            
            # Find the index of the row with the minimum and maximum value in the 'u_pu' column
            max_index = line['loading'].idxmax()
            min_index = line['loading'].idxmin()

            # Retrieve the row with the minimum and maximum value in the 'u_pu' column
            max_row = line.loc[max_index]
            min_row = line.loc[min_index]
            
            # Put the data in the correct rows, and columns
            df_line_loss.loc[id, 'energy_loss'] = energy_loss
            df_line_loss.loc[id, 'timestamp_max'] = max_row['Timestamp']
            df_line_loss.loc[id, 'max_loading'] = max_row['loading']
            df_line_loss.loc[id, 'timestamp_min'] = min_row['Timestamp']
            df_line_loss.loc[id, 'min_loading'] = min_row['loading']

        print(df_line_loss)

    def test_see_output(self, dir):
        # self.test = pd.read_parquet(dir)
        # print(self.test)
        pass
        

