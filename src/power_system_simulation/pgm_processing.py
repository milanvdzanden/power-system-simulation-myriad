"""
Implementation of a power grid simulation and calculation module using the power-grid-model package as the core. Additional functions are included to display the data. 
"""
import json
import pprint
import warnings

with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
    # suppress warning about pyarrow as future required dependency
    from pandas import DataFrame
    
import pandas as pd
import power_grid_model as pgm
from power_grid_model.utils import json_deserialize, json_serialize

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
        # pprint.pprint(json.loads(data))
        dataset = json_deserialize(data)

        # print("components:", dataset.keys())
        # f = open(dir_network_json, "r")
        # print(f.read())
        
        # Read active and reactive load profile from parquet file
        self.active_load_profile = pd.read_parquet(dir_active_profile)
        self.reactive_load_profile = pd.read_parquet(dir_reactive_profile)
        pass

    def create_update_model(self):
        """
        How we think this works is that from the input data (that is read in __init__), you create an update profile that changes the load profiles
        for each timestamp, which allows the batch calculation to run directly.

        Alternatively, it may be also needed that for the batch, you need to "update" on each timestamp. In that case, this function could be made private
        and called between each calculation of the power flow.
        Store results in a self.[...] variable        
        """        
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


