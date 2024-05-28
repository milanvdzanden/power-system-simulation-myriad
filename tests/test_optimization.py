import os
import sys
import pytest
import pandas as pd

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests/tests_data/A3"))
sys.path.append(src_dir)
src_dir = src_dir.replace("\\", "/")

import networkx as nx
import power_system_simulation.optimization as psso
from power_grid_model.utils import json_deserialize, json_serialize

import json

def test_optimization():
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

test_optimization()    
    
    
    