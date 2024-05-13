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
    
import pandas as pd
import power_grid_model as pgm
from power_grid_model.utils import json_deserialize, json_serialize
from power_grid_model.validation import ValidationException

# ValidationError: included in package

class ProfilesDontMatchError(Exception):
    pass

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
        try:
            self.pgm = pgm.PowerGridModel(self.pgm_input)
        except pgm.validation.assert_valid_input_data(self.pgm):
            raise ValidationException
        
        pass

    def create_update_model(self):
        """
        How we think this works is that from the input data (that is read in __init__), you create an update profile that changes the load profiles
        for each timestamp, which allows the batch calculation to run directly.

        Alternatively, it may be also needed that for the batch, you need to "update" on each timestamp. In that case, this function could be made private
        and called between each calculation of the power flow.
        Store results in a self.[...] variable        
        """   
        # print(self.pgm_input["node"][1])  
        # print(self.pgm_input["line"]) 
        # print(self.pgm_input["sym_load"])   
        # print(self.pgm_input["source"])  
        print(self.active_load_profile)
        print(self.reactive_load_profile)
        
        # Check if rows and colums of active and reactive profiles match
        if self.active_load_profile.shape != self.reactive_load_profile.shape:
            raise ProfilesDontMatchError("Number of rows and columns in active and reactive profiles do not match.")
        
        # Check if load ids and timestamps of active and reactive profiles match
        if not self.active_load_profile.equals(self.reactive_load_profile):
            raise ProfilesDontMatchError("Timestamps and load ids in active and reactive profiles do not match.")
        
        pass

    def run_batch_process(self):
        """
        Run the batch process on the input data according to the load update profile.
        Store results in a self.[...] variable
        """        
        pass

    def get_aggregate_results(self) -> list[pd.DataFrame, pd.DataFrame]:
        """
        Generate the two required output tables based on the self variables created in run_batch_process
        Output is a 2-element list of data frames for the 2 required aggregated tables
        """        
        pass
        

