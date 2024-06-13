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

    # Using active profile data from A3 as "wrong" input for this network
    dir_active_profile_wrong = src_dir + "/active_power_profile_wrong.parquet"
    active_load_profile_wrong = pd.read_parquet(dir_active_profile_wrong)
    with pytest.raises(pgm_p.PgmProcessor.ProfilesDontMatchError(0)) as excinfo:
        p = pgm_p.PgmProcessor(network_data, active_load_profile_wrong, reactive_load_profile)
        p.create_update_model()


test_pgm_processing()
