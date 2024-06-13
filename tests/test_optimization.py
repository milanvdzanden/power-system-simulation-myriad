import os
import sys
import pytest
import pandas as pd
import numpy as np

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests/tests_data/A3"))
sys.path.append(src_dir)
src_dir = src_dir.replace("\\", "/")

import networkx as nx
import power_system_simulation.optimization as psso
from power_grid_model.utils import json_deserialize, json_serialize

import json
import copy as copy

meta_data = {}
network_data = {}
active_profile = {}
reactive_profile = {}
ev_active_profile = {}

def test_optimization():
    global network_data, meta_data, active_profile, reactive_profile, ev_active_profile
    dir_meta_data_json = src_dir + "/meta_data.json"
    dir_network_json = src_dir + "/input_network_data.json"
    dir_active_profile = src_dir + "/active_power_profile.parquet"
    dir_reactive_profile = src_dir + "/reactive_power_profile.parquet"
    dir_ev_active_profile = src_dir + "/ev_active_power_profile.parquet"
    
    
    with open(dir_meta_data_json) as fp:
        data = fp.read()
    meta_data = json.loads(data)
    
    with open(dir_network_json) as fp:
        data = fp.read()
    network_data = json_deserialize(data)
    
    active_profile = pd.read_parquet(dir_active_profile)
    reactive_profile = pd.read_parquet(dir_reactive_profile)
    ev_active_profile = pd.read_parquet(dir_ev_active_profile)
    

    
    """
    Insert here the file that will be given. (not available yet)
    Choose a random seed to keep it consistent.
    
    Testing every error.
    
    EV penetration:
        Check if number  of EVs per LV feeder are correct in the end.
        Check random penetration levels, also if above 100% etc.
        
    Optimal tap position:
        Just a print.
    
    N-1 calculation:
        Check if the given Line ID is not a valid line.
        Check if given Line ID is not connected at both sides in the base case.
        Check the table if there are no alternatives.
        Just checking if it works with different line IDs.
    
    """
    
    p = psso.LV_grid(network_data, active_profile, reactive_profile, ev_active_profile, meta_data)
    p.N_1_calculation(18)
    p.EV_penetration_level(0.8)

def test_errors():
    global network_data, meta_data, active_profile, reactive_profile, ev_active_profile
    
    #2 The LV grid has exactly one transformer, and one source
    #2.1 extra transformer
    test_2_transformer = copy.deepcopy(network_data)
    transformer2 = copy.deepcopy(test_2_transformer["transformer"][0])
    transformer2["id"] = 25
    test_2_transformer["transformer"] = np.append(test_2_transformer["transformer"],transformer2)
    with pytest.raises(psso.LvGridOneTransformerAndSource) as excinfo:
        psso.LV_grid(
            test_2_transformer, active_profile, reactive_profile, ev_active_profile, meta_data
        )
    #2.2 extra source
    test_2_source = copy.deepcopy(network_data)
    source2 = copy.deepcopy(test_2_source["source"][0])
    source2["id"] = 25
    test_2_source["source"] = np.append(test_2_source["source"],source2)
    with pytest.raises(psso.LvGridOneTransformerAndSource) as excinfo:
        psso.LV_grid(
            test_2_source, active_profile, reactive_profile, ev_active_profile, meta_data
        )
    #3 All IDs in the LV Feeder IDs are valid line IDs.
    test_valid_ids = copy.deepcopy(network_data)
    test_valid_ids["line"]["id"][0] = 25
    
    with pytest.raises(psso.LVFeederError, match=r".*T0") as excinfo:
        psso.LV_grid(
            test_valid_ids, active_profile, reactive_profile, ev_active_profile, meta_data
        )
    #4 All the lines in the LV Feeder IDs have the from_node the same as the to_node of the transformer.
    test_same_nodes = copy.deepcopy(network_data)
    test_same_nodes["transformer"]["to_node"] = 2
    
    with pytest.raises(psso.LVFeederError, match=r".*T1") as excinfo:
        psso.LV_grid(
            test_same_nodes, active_profile, reactive_profile, ev_active_profile, meta_data
        )
    #5 The grid is fully connected in the initial state. 
    # test_fully_connected = copy.deepcopy(network_data)
    # test_fully_connected["line"]["to_status"][1] = 0
    
    # print(test_fully_connected["line"])
    # with pytest.raises(psso.GraphNotFullyConnectedError) as excinfo:
    #     psso.LV_grid(
    #         test_fully_connected, active_profile, reactive_profile, ev_active_profile, meta_data
    #     )
    #6 The grid has no cycles in the initial state.
    test_cycles = copy.deepcopy(network_data)
    test_cycles["line"]["to_status"][8] = 1
    
    with pytest.raises(psso.GraphCycleError) as excinfo:
        psso.LV_grid(
            test_cycles, active_profile, reactive_profile, ev_active_profile, meta_data
        )
    #7 The timestamps are matching between the active load profile, reactive load profile, and EV charging profile.
    # test_timestamps = copy.deepcopy(active_profile)
    # test_timestamps.index = test_timestamps.index.map(lambda x: x.replace(year=2026))
    # with pytest.raises(psso.ProfilesDontMatchError, match=r".*T0") as excinfo:
    #     psso.LV_grid(
    #         network_data, test_timestamps, reactive_profile, ev_active_profile, meta_data
    #     )
    #8 The IDs in active load profile and reactive load profile are matching.
    # test_load_ids = copy.deepcopy(active_profile)
    # print(test_load_ids.columns)
    # test_load_ids.columns = np.array([25, 13, 14, 15], dtype='int32')
    # test_load_ids.columns.name = 'Load ID'
    # print(test_load_ids.columns)
    # with pytest.raises(psso.ProfilesDontMatchError, match=r".*T1") as excinfo:
    #     psso.LV_grid(
    #         network_data, test_load_ids, reactive_profile, ev_active_profile, meta_data
    #     )
        
test_optimization()    
test_errors()  
    
    