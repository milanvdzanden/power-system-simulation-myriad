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
import math as math   
import copy
import random
import numpy as np
import pandas as pd
import power_grid_model as pgm
from power_grid_model.utils import json_deserialize, json_serialize
from power_grid_model.validation import ValidationException
from power_grid_model import *

from power_grid_model import LoadGenType
from power_grid_model import PowerGridModel, CalculationMethod, CalculationType, MeasuredTerminalType
from power_grid_model import initialize_array

import power_system_simulation.graph_processing as pss
import power_system_simulation.pgm_processing as pgm_p



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
pass
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
pass

class LV_grid:

    """
    DOCSTRING
    """
    
    def __init__(self, network_data: str, active_profile: str, reactive_profile: str, ev_active_profile: str, meta_data: str):   
         
        """
        Serializing the json file.
        """ 
        self.meta_data = meta_data
        
        self.pgm_input = network_data
        # Read active and reactive load profile from parquet file
        self.active_load_profile = active_profile
        
        self.reactive_load_profile = reactive_profile
        
        self.ev_active_profile = ev_active_profile
        
        pgm.validation.assert_valid_input_data(self.pgm_input)

        self.pgm_model = pgm.PowerGridModel(self.pgm_input)
        
        
        """
        Validity checks need to be made to ensure that there is no overlap or mismatching between the relevant IDs and profiles. Read 'input validity check' in assignment 3 for the specific checks.
        also raise or passthrough relevant errors.
        """
        
        
        """
        NOTE: the funtionalities are independent from each other. For example, for optimal tap position analysis, you need to analyse the original grid with house profile, WITHOUT the EV profile.
        """
        pass

    def optimal_tap_position(self,optimization_criteri: str): 
        
        """
        Optimize the tap position of the transformer in the LV grid.
        Run a one time power flow calculation on every possible tap postition for every timestamp (https://power-grid-model.readthedocs.io/en/stable/examples/Transformer%20Examples.html).
        The most opmtimized tap position should have the min total energy loss of all lines and whole period and min. deviation of p.u. node voltages w.r.t. 1 p.u.
        (We think that total energy loss has more importance than the Delta p.u.)
        The user can choose the criteria for optimization, so thye can choose how low the energy_loss and voltage_deviation should be for it to be valid.
    
        Args:
            optimization_criteria (str): Criteria for optimization (e.g., 'energy_loss', 'voltage_deviation').

        Returns:
            tuple: Optimal tap position and corresponding performance metrics.
        """
        pass
    def EV_penetration_level(self,penetration_level : float): 
        
        """
        Randomly adding EV charging profiles according to a couple criterea using the input 'penetration_level'.
        
        First: The number of EVs per LV feeder needs to be equal to 'round_down[penetration_level * total_houses / number_of_feeders]'
        To our understanding, an EV feeder is just a branch. This means that the list of feeder IDs contains all line IDs of every start of a branch.
        
        Second: Use Assignment 1 to see which house is in which branch/feeder.
        Then within each feeder randomly select houses which will have an EV charger. Don't forget the number of EVs per LV feeder.
                    
        Third: For each selected house with EV, randomly select an EV charging profile to add to the sym_load of that house.
        Every profile can not be used twice -> there will be enough profiles to cover all of sym_load.
                    
        Last: Get the two aggregation tables using Assignment 2 and return these.

        Args:
            penetration_level (float): Percentage of houses with an Electric Vehicle (EV) charger. -> user input

        Returns:
            Two aggregation tables using Assignment 2.
            
        NOTE: The EV charging profile does not have sym_load IDs in the column header. They are just sequence numbers of the pool. Assigning the EV profiles to sym_load is part of the assignment tasks.
        """        
        #make Graphprocessing instance to randomly select houses with ev         
        vertex_ids = [node[0] for node in self.pgm_input["node"]]
        edge_ids = [edge[0] for edge in self.pgm_input["line"]]
        edge_vertex_id_pairs = [(edge[1], edge[2]) for edge in self.pgm_input["line"]]
        edge_enabled = [(edge[3] == 1 and edge[4] == 1) for edge in self.pgm_input["line"]]
        source_vertex_id = self.pgm_input["source"][0][1]
        
        # Pretend all transformers are short circuits, so that GraphProcessor can use it
        for transformer in self.pgm_input["transformer"]:
            edge_vertex_id_pairs.append((transformer[1], transformer[2]))
            edge_enabled.append(True)
            edge_ids.append(transformer[0])
        gp = pss.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
        
        #use the instance to know which houses for which feeder
        #see which lines are feeders and which nodes/houses are connected
        
        #random.seed(0)
        feeder_nodes = {}
        feeders = []
        feeder_houses = {}
        total_real_house_per_LV = []
        EV_houses = {}
        House_Profile = {}
        
        #see which feeder has which nodes
        for feeder_id in self.meta_data["lv_feeders"]:
            feeders.append(feeder_id)
            feeder_nodes[feeder_id] = gp.find_downstream_vertices(feeder_id)
        
        #get the total houses and nmr lv feeders from pgm_input
        sym_houses = [house[1] for house in self.pgm_input["sym_load"]]
        total_houses = len(sym_houses)
        total_feeders = len(feeder_nodes)
        nmr_ev_per_lv_feeder = math.floor(penetration_level * total_houses / total_feeders)
        
        #get which houses are from which feeder
        for x in range(len(feeders)):
            houses = [i for i in sym_houses if i in feeder_nodes[feeders[x]]]
            total_real_house_per_LV.append(houses)
        for x in range(len(feeders)):
            feeder_id = self.meta_data["lv_feeders"][x % len(self.meta_data["lv_feeders"])]
            feeder_houses[feeder_id] = total_real_house_per_LV[x]
        
        #get which houses will have an EV per feeder
        for feeder_id in self.meta_data["lv_feeders"]:
            random_houses_ev = random.sample(feeder_houses[feeder_id], nmr_ev_per_lv_feeder)
            EV_houses[feeder_id] = random_houses_ev    
        parquet_df = pd.DataFrame(self.ev_active_profile)
        columns = list(parquet_df.columns.values)
        
        #get random profiles with no repetitives
        random.shuffle(columns)
        amount_profiles = len(feeders)
        random_profile = [columns[i:i+nmr_ev_per_lv_feeder] for i in range(0, len(columns), nmr_ev_per_lv_feeder)][:amount_profiles]
        
        #assign which random profile is paired with which house
        index_of_feeders = 0
        for x in feeders:
            index_of_random_profile = 0
            for i in EV_houses[x]:
                House_number_list = EV_houses[x]
                House_number = House_number_list[index_of_random_profile]
                House_Profile[House_number] = random_profile[index_of_feeders][index_of_random_profile]
                index_of_random_profile = index_of_random_profile+1    
            index_of_feeders = index_of_feeders+1
        
        
    def N_1_calculation(self,line_id):
        
        """
        One line will be disconnected -> generate alternative grid topoligy.
        Check and raise the relevant errors.
        Find a list of IDs that are now disconnected, but can be connected to make the grid fully connected. (Literally find_alternative_edges from Assignment 1).
        Run the time-series power flow calculation for the whole time period for every alternative connected line. (https://power-grid-model.readthedocs.io/en/stable/examples/Transformer%20Examples.html)
        Return a table to summarize the results with the relevant columns stated in the assignment (Every row of this table is a scenario with a different alternative line connected.)
        If there are no alternatives, it should return an emtpy table with the correct format and titles in the columns and rows.

        Args:
            line_id (int): The uses will give a line_id that will be disconnected.

        Returns:
            Table with results of every scenario with a different alternative line connected.
        """
        vertex_ids = [node[0] for node in self.pgm_input["node"]]
        edge_ids = [edge[0] for edge in self.pgm_input["line"]]
        edge_vertex_id_pairs = [(edge[1], edge[2]) for edge in self.pgm_input["line"]]
        edge_enabled = [(edge[3] == 1 and edge[4] == 1) for edge in self.pgm_input["line"]]
        source_vertex_id = self.pgm_input["source"][0][1]
        
        # Pretend all transformers are short circuits, so that GraphProcessor can use it
        for transformer in self.pgm_input["transformer"]:
            edge_vertex_id_pairs.append((transformer[1], transformer[2]))
            edge_enabled.append(True)
            edge_ids.append(transformer[0])
        
        gp = pss.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
        
        for alternative_edge in gp.find_alternative_edges(line_id):
            # Enable the alternative edge in the json, disable line_id (the input line)
            pgm_input_alternative = copy.deepcopy(self.pgm_input)
            alternative_edge_index = next((i for i, item in enumerate(pgm_input_alternative['line']) if item['id'] == alternative_edge), None)
            input_edge_index = next((i for i, item in enumerate(pgm_input_alternative['line']) if item['id'] == line_id), None)
            
            # Enable the alternative edge
            pgm_input_alternative['line'][alternative_edge_index][4] = 1
            pgm_input_alternative['line'][alternative_edge_index][3] = 1
            
            # Disable the input edge
            pgm_input_alternative['line'][input_edge_index][4] = 0
            pgm_input_alternative['line'][input_edge_index][3] = 0
            
            # Do the calculation for pgm_input_alternative
            #p = pgm_p.PgmProcessor()
            
