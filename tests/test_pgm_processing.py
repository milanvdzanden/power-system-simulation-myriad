import os
import sys
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests/tests_data"))
sys.path.append(src_dir)
src_dir = src_dir.replace("\\", "/")

import networkx as nx
import power_system_simulation.pgm_processing as pgm_p

def test_pgm_processing():
    dir_network_json = src_dir + "/input_network_data.json"
    dir_active_profile = src_dir + "/active_power_profile.parquet"
    dir_reactive_profile = src_dir + "/reactive_power_profile.parquet"

    p = pgm_p.PgmProcessor(dir_network_json, dir_active_profile, dir_reactive_profile)

    # Process p here and get output data tables

test_pgm_processing()