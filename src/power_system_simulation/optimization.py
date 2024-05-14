"""
Building a package with low voltage grid analytics functions.
"""
from typing import List, Tuple
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
from power_grid_model import *

from power_grid_model import LoadGenType
from power_grid_model import PowerGridModel, CalculationMethod, CalculationType, MeasuredTerminalType
from power_grid_model import initialize_array



class LV_Feeder_error(Exception):

    """
    Error class for the LV feeder.
    """
    def __init__(self, mode):
        """
        Prints the exception message using the standard exception class.

        Args:
          self: passes exception
          mode: error type,
            0 if the IDs are not valid
            1 if from_node is not the same node as to_node
        """    

class Load_profile_error(Exception):
    
    """
    Error class for the Load_profile.
    """    
    
    def __init__(self, mode):
        """
        Prints the exception message using the standard exception class.

        Args:
          self: passes exception
          mode: error type,
            0 if the time stamps are not matching between the active load profile, reactive load profile, and EV charging profile.
            1 if the node ids do not match each other in the profiles.
            2 if the IDs in the profiles are not valid IDs of sym_load.
        """


class LV_grid:

    """
    DOCSTRING
    """
    
    def __init__(self, dir_meta_data_json: str, dir_active_profile: str, dir_reactive_profile: str, dir_EV_charging_profile: str, LV_feeder_ids: List[int]):   
         
        """
        Serializing the json file.
        """ 
        with open(dir_meta_data_json) as fp:
            data = fp.read()
        dataset = json_deserialize(data)
        self.pgm_input = dataset
        
        # Read active and reactive load profile from parquet file
        self.active_load_profile = pd.read_parquet(dir_active_profile)
        self.reactive_load_profile = pd.read_parquet(dir_reactive_profile)
        
        pgm.validation.assert_valid_input_data(self.pgm_input)

        self.pgm_model = pgm.PowerGridModel(self.pgm_input)
        
        """
        The following validity checks need to be ran:
        
        - The LV grid should be a valid PGM input data.
        - The LV grid has exactly one transformer, and one source.
        - All IDs in the LV Feeder IDs are valid line IDs.
        - All the lines in the LV Feeder IDs have the from_node the same as the to_node of the transformer.
        - The grid is fully connected in the initial state.
        - The grid has no cycles in the initial state.
        - The timestamps are matching between the active load profile, reactive load profile, and EV charging profile.
        - The IDs in active load profile and reactive load profile are matching.
        - The IDs in active load profile and reactive load profile are valid IDs of sym_load.
        - The number of EV charging profile is at least the same as the number of sym_load.
        """
        
        
        """
        NOTE: the funtionalities are independent from each other. For example, for optimal tap position analysis, you need to analyse the original grid with house profile, WITHOUT the EV profile.
        """


    def optimal_tap_position(self,optimization_criteri: str): 
        
        """
        Optimize the tap position of the transformer in the LV grid.
        Run a one time power flow calculation on every possible tap postition for every timestamp (https://power-grid-model.readthedocs.io/en/stable/examples/Transformer%20Examples.html).
        The most opmtimized tap position has:
           - The minimal total energy loss of all the lines and the whole time period.
           - The minimal Delta p.u. averaged across all the nodes with respect to 1 p.u.
        (We think that total energy loss has more imprtance than the Delta p.u.)
        The user can choose the criterea for optimization, so thye can choose how low the energy_loss and voltage_deviation should be for it to be valid.
    
        Args:
            optimization_criteria (str): Criteria for optimization (e.g., 'energy_loss', 'voltage_deviation').

        Returns:
            tuple: Optimal tap position and corresponding performance metrics.
        """
    
    def EV_penetration_level(self,penetration_level : float): 
        
        """
        Randomly adding EV charging profiles according to a couple criterea using the input 'penetration_level'.
        First: The number of EVs per LV feeder needs to be equal to 'round_down[penetration_level * total_houses / number_of_feeders]'
        To our understanding, an EV feeder is just a branch. This means that the list of feeder IDs contains all line IDs of every start of a branch.
        Second: Use Assignment 1 to see which house is in which branch/feeder.
                    Then within each feeder randomly select houses whih will have an EV charger. Don't forget the number of EVs per LV feeder.
        Third: For each selected house with EV, randomly select an EV charging profile to add to the sym_load of that house.
                    Every profile can not be used twice -> there will be enough profiles to cover all of sym_load.
        Last: Get the two aggregation tables using Assignment 2 and return these.

        Args:
            penetration_level (float): Percentage of houses with an Electric Vehicle (EV) charger. -> user input

        Returns:
            Two ggregation tables using Assignment 2.
            
        NOTE: The EV charging profile does not have sym_load IDs in the column header. They are just sequence numbers of the pool. Assigning the EV profiles to sym_load is part of the assignment tasks.
        """
    def N_1_calculation(self,line_id):
        
        """
        One line will be disconnected -> generate alternative grid topoligy.
        Check if the given Line_ID is valid, if not raise error.
        Check if the given Line_ID is connected at both sides, if not raise error.
        Find a list of IDs that are now disconnected, but can be connected to make the grid fully connected. (Literally find_alternative_edges from Assignment 1).
        Run the time-series power flow calculation for the whole time period for every alternative connected line. (https://power-grid-model.readthedocs.io/en/stable/examples/Transformer%20Examples.html)
        Return a table with the results witht he following columns:
            - The alternative Line ID to be connected
            - The maximum loading among of lines and timestamps
            - The Line ID of this maximum
            - The timestamp of this maximum
        Every row of this table is a scenario with a different alternative line connected.
        
        If there are no alternatives, it should return an emtpy table with the correct format and titles in the columns and rows.

        Args:
            line_id (int): The uses will give a line_id that will be disconnected.

        Returns:
            Table with results of every scenario with a different alternative line connected.
        """
        
