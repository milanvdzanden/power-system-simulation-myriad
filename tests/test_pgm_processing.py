import os
import sys

import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests/tests_data/A2"))
sys.path.append(src_dir)
src_dir = src_dir.replace("\\", "/")

import networkx as nx
import pandas as pd
from power_grid_model.utils import json_deserialize, json_serialize

import power_system_simulation.pgm_processing as pgm_p

def test_pgm_processing():
    dir_network_json = src_dir + "/input_network_data.json"
    dir_active_profile = src_dir + "/active_power_profile.parquet"
    dir_reactive_profile = src_dir + "/reactive_power_profile.parquet"

    with open(dir_network_json) as fp:
        data = fp.read()
    network_data = json_deserialize(data)

    # Read active and reactive load profile from parquet file
    active_load_profile = pd.read_parquet(dir_active_profile)
    reactive_load_profile = pd.read_parquet(dir_reactive_profile)

    p = pgm_p.PgmProcessor(network_data, active_load_profile, reactive_load_profile)
    p.create_update_model()
    p.run_batch_process()
    aggregate_results = p.get_aggregate_results()
    
    # Save aggregate results (for drawing tests in external notebook as repository is not completed yet and cannot be used directly as-is)
    aggregate_results[0].to_parquet(src_dir + "/calculated_output_per_timestamp.parquet")
    aggregate_results[1].to_parquet(src_dir + "/calculated_output_per_line.parquet")

    # Change a timestamp in active profile to an incorrect one and check for error
    active_load_profile_wrong = active_load_profile.copy()
    active_load_profile_wrong.rename(index={active_load_profile.index[0]:pd.to_datetime('today').normalize()}, inplace=True)
    with pytest.raises(pgm_p.ProfilesDontMatchError, match=r".*T0") as excinfo:
        p = pgm_p.PgmProcessor(network_data, active_load_profile_wrong, reactive_load_profile)
        p.create_update_model()

    # Change a node ID in active profile to an incorrect one and check for error
    active_load_profile_wrong = active_load_profile.copy()
    active_load_profile_wrong.rename(columns={active_load_profile.columns[0]: 1234567890}, inplace=True)
    with pytest.raises(pgm_p.ProfilesDontMatchError, match=r".*T1") as excinfo:
        p = pgm_p.PgmProcessor(network_data, active_load_profile_wrong, reactive_load_profile)
        p.create_update_model()

    # Change a node ID in the network description to an incorrect one (different than reactive/active profile) and check for error
    network_data_wrong = network_data.copy()
    network_data_wrong['sym_load'][0][0] = 1234567890
    with pytest.raises(pgm_p.ProfilesDontMatchError, match=r".*T2") as excinfo:
        p = pgm_p.PgmProcessor(network_data_wrong, active_load_profile, reactive_load_profile)
        p.create_update_model()

    # Test if pre-calculated output data matches the aggregated output
    dir_out_per_line = src_dir + "/output_table_row_per_line.parquet"
    dir_out_per_timestamp = src_dir + "/output_table_row_per_timestamp.parquet"

    assert (
        p.compare_to_expected(
            aggregate_results,
            pd.read_parquet(dir_out_per_line),
            pd.read_parquet(dir_out_per_timestamp),
        )
        == True
    )

    # Change the output data and check if the error is detected
    with pytest.raises(AssertionError) as excinfo:
        assert (
            p.compare_to_expected(
                aggregate_results,
                pd.read_parquet(dir_out_per_line),
                pd.read_parquet(dir_out_per_line),
            )
            == True
        )

test_pgm_processing()
