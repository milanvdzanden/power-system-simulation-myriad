import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from power_system_simulation.simple_function import add


def test_add():
    assert add(1, 1) == 2


test_add()
