import os
import sys

import numpy as np
import pandas as pd
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests/tests_data/A3"))
sys.path.append(src_dir)
src_dir = src_dir.replace("\\", "/")

import copy as copy
import json

import networkx as nx
from power_grid_model.utils import json_deserialize, json_serialize

import power_system_simulation.graph_processing as pss
import power_system_simulation.optimization as psso

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
    p.n_1_calculation(18)

    # Test EV penetration level; Obtain aggregated results
    aggregate_results = p.EV_penetration_level(0.8, True)

    # Save aggregate results (for drawing tests in external notebook)
    aggregate_results[0].to_parquet(src_dir + "/calculated_output_per_timestamp.parquet")
    aggregate_results[1].to_parquet(src_dir + "/calculated_output_per_line.parquet")

    # Test optimal tap position
    p.optimal_tap_position("energy_loss")
    p.optimal_tap_position("voltage_deviation")
    # Test optimal tap position, with wrong directive (should return -1, -1)
    assert set(p.optimal_tap_position("wrong_directive")) == set([-1, -1])


def test_errors():
    global network_data, meta_data, active_profile, reactive_profile, ev_active_profile

    # 2 The LV grid has exactly one transformer, and one source
    # 2.1 extra transformer
    test_2_transformer = copy.deepcopy(network_data)

    transformer2 = copy.deepcopy(test_2_transformer["transformer"][0])
    transformer2["id"] = 25
    test_2_transformer["transformer"] = np.append(test_2_transformer["transformer"], transformer2)
    with pytest.raises(psso.LvGridOneTransformerAndSource) as excinfo:
        psso.LV_grid(
            test_2_transformer, active_profile, reactive_profile, ev_active_profile, meta_data
        )
    # 2.2 extra source
    test_2_source = copy.deepcopy(network_data)
    source2 = copy.deepcopy(test_2_source["source"][0])
    source2["id"] = 25
    test_2_source["source"] = np.append(test_2_source["source"], source2)
    with pytest.raises(psso.LvGridOneTransformerAndSource) as excinfo:
        psso.LV_grid(test_2_source, active_profile, reactive_profile, ev_active_profile, meta_data)

    # 3 All IDs in the LV Feeder IDs are valid line IDs.
    test_valid_ids = copy.deepcopy(network_data)
    test_valid_ids["line"]["id"][0] = 25

    with pytest.raises(psso.LVFeederError, match=r".*T0") as excinfo:
        psso.LV_grid(test_valid_ids, active_profile, reactive_profile, ev_active_profile, meta_data)
    # 4 All the lines in the LV Feeder IDs have the from_node the same as the to_node of the transformer.
    test_same_nodes = copy.deepcopy(network_data)
    test_same_nodes["transformer"]["to_node"] = 2

    with pytest.raises(psso.LVFeederError, match=r".*T1") as excinfo:
        psso.LV_grid(
            test_same_nodes, active_profile, reactive_profile, ev_active_profile, meta_data
        )
    # 5 The grid is fully connected in the initial state.
    test_fully_connected = copy.deepcopy(network_data)
    test_fully_connected["line"]["to_status"][4] = 0
    test_fully_connected["line"]["from_status"][5] = 0

    with pytest.raises(pss.GraphNotFullyConnectedError) as excinfo:
        psso.LV_grid(
            test_fully_connected, active_profile, reactive_profile, ev_active_profile, meta_data
        )
    # 6 The grid has no cycles in the initial state.
    test_cycles = copy.deepcopy(network_data)
    test_cycles["line"]["to_status"][8] = 1

    with pytest.raises(pss.GraphCycleError) as excinfo:
        psso.LV_grid(test_cycles, active_profile, reactive_profile, ev_active_profile, meta_data)

    # 7 The timestamps are matching between the active load profile, reactive load profile, and EV charging profile.
    active_load_profile_wrong_time = copy.deepcopy(active_profile)
    active_load_profile_wrong_time.rename(
        index={active_profile.index[0]: pd.to_datetime("today").normalize()}, inplace=True
    )
    with pytest.raises(psso.ProfilesDontMatchError, match=r".*T0") as excinfo:
        psso.LV_grid(
            network_data,
            active_load_profile_wrong_time,
            reactive_profile,
            ev_active_profile,
            meta_data,
        )

    # 8 The IDs in active load profile and reactive load profile are matching.
    active_load_profile_wrong_ID = copy.deepcopy(active_profile)
    active_load_profile_wrong_ID.rename(
        columns={active_profile.columns[0]: 1234567890}, inplace=True
    )
    with pytest.raises(psso.ProfilesDontMatchError, match=r".*T1") as excinfo:
        psso.LV_grid(
            network_data,
            active_load_profile_wrong_ID,
            reactive_profile,
            ev_active_profile,
            meta_data,
        )

    # 9 The IDs in active load profile and reactive load profile are valid IDs of sym_load.
    test_sym_load = copy.deepcopy(network_data)
    test_sym_load["sym_load"]["id"][0] = 28
    with pytest.raises(psso.ProfilesDontMatchError, match=r".*T2") as excinfo:
        psso.LV_grid(test_sym_load, active_profile, reactive_profile, ev_active_profile, meta_data)

    # 10 The number of EV charging profile is at least the same as the number of sym_load.
    test_number_EV = copy.deepcopy(ev_active_profile)
    new_column = copy.deepcopy(test_number_EV[1])
    test_number_EV[4] = new_column
    with pytest.raises(psso.EvProfilesDontMatchSymLoadError) as excinfo:
        psso.LV_grid(network_data, active_profile, reactive_profile, test_number_EV, meta_data)


test_optimization()
test_errors()
