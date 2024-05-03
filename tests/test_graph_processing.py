import os
import sys
import time

import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

start_time = time.time()

import power_system_simulation.graph_processing as pss


def test_graph_processing():
    # edge_id = 10
    vertex_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    edge_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    edge_vertex_id_pairs = [
        (2, 6),
        (1, 2),
        (2, 3),
        (3, 4),
        (6, 7),
        (6, 9),
        (4, 5),
        (7, 8),
        (9, 10),
        (1, 11),
    ]
    edge_enabled = [True, True, True, True, True, True, True, False, True, True]
    source_vertex_id = 1
    gp = pss.GraphProcessor(
        vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id
    )
    with pytest.raises(pss.IDNotFoundError) as excinfo:
        gp.find_downstream_vertices(12)

    assert set(gp.find_downstream_vertices(1)) == set([6, 7, 9, 10])
    assert set(gp.find_downstream_vertices(2)) == set([2, 3, 4, 5, 6, 7, 9, 10])
    assert set(gp.find_downstream_vertices(3)) == set([3, 4, 5])
    assert set(gp.find_downstream_vertices(4)) == set([4, 5])
    assert set(gp.find_downstream_vertices(5)) == set([7])
    assert set(gp.find_downstream_vertices(6)) == set([9, 10])
    assert set(gp.find_downstream_vertices(7)) == set([5])
    assert set(gp.find_downstream_vertices(8)) == set([])  # Edge 8 is disabled
    assert set(gp.find_downstream_vertices(9)) == set([10])
    assert set(gp.find_downstream_vertices(10)) == set([11])

    """
    v1 ----- edge2 ------ v2 ----- edge3 ------ v3 ----- edge4 ------ v4 ----- edge7 ------ v5
    |                      |
  edge10                 edge1
    |                      |                          (disabled)
   v11                    v6 ----- edge5 ------ v7 ----- edge8 ------ v8
                           |
                         edge6
                           |
                          v9 ----- edge9 ------ v10
    """

    # print(gp.find_downstream_vertices(edge_id))


test_graph_processing()

print("Process finished --- %s seconds ---" % (time.time() - start_time))
