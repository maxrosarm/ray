import json
import platform
import random
import re
import shutil
import sys
import time
import zlib
from collections import defaultdict

import numpy as np
import pytest

import ray
from ray._private.test_utils import wait_for_condition
from ray.cluster_utils import Cluster, cluster_not_supported
from ray.tests.test_object_spilling import assert_no_thrashing, is_dir_empty

# Note: Disk write speed can be as low as 6 MiB/s in AWS Mac instances, so we have to
# increase the timeout.
pytestmark = [pytest.mark.timeout(900 if platform.system() == "Darwin" else 180)]

@pytest.mark.skipif(True, reason="Not needed at the moment")
def _check_spilled(num_objects_spilled=0):
    def ok():
        s = ray._private.internal_api.memory_summary(stats_only=True)
        if num_objects_spilled == 0:
            return "Spilled " not in s

        m = re.search(r"Spilled (\d+) MiB, (\d+) objects", s)
        if m is not None:
            actual_num_objects = int(m.group(2))
            return actual_num_objects >= num_objects_spilled

        return False

    wait_for_condition(ok, timeout=90, retry_interval_ms=5000)

# have to figure out which node spills what and which pulls 

def test_pull_spilled_object(
    ray_start_cluster_enabled, multi_node_object_spilling_config, shutdown_only
):
    cluster = ray_start_cluster_enabled
    object_spilling_config, _ = multi_node_object_spilling_config

    # Head node.
    cluster.add_node(
        num_cpus=1,
        resources={"pool_id_0": 1},
        object_store_memory=75 * 1024 * 1024,
        _system_config={
            "max_io_workers": 2,
            "min_spilling_size": 1 * 1024 * 1024,
            "automatic_object_spilling_enabled": True,
            "object_store_full_delay_ms": 100,
            "object_spilling_config": object_spilling_config,
        },
    )
    ray.init(cluster.address)

    # add 1 worker node
    cluster.add_node(
        num_cpus=1, resources={"pool_id_1": 1}, object_store_memory=75 * 1024 * 1024
    )
    cluster.wait_for_nodes()

    # create the objects on the remote node?
    @ray.remote(num_cpus=1, resources={"pool_id_1": 1})
    def create_objects():
        results = []
        for size in range(5):
            arr = np.random.rand(size * 1024 * 1024)
            hash_value = zlib.crc32(arr.tobytes())
            results.append([ray.put(arr), hash_value])
        # ensure the objects are spilled
        arr = np.random.rand(5 * 1024 * 1024)
        ray.get(ray.put(arr))
        ray.get(ray.put(arr))
        return results

    # get the objects and place into head node?
    @ray.remote(num_cpus=1, resources={"pool_id_0": 1})
    def get_object(arr):
        return zlib.crc32(arr.tobytes())

    # create the objects on the remote node?
    results = ray.get(create_objects.remote())

    # restore on remote node and send over to head node?
    for value_ref, hash_value in results:
        hash_value1 = ray.get(get_object.remote(value_ref))
        assert hash_value == hash_value1

if __name__ == "__main__":
    import os
    if(True):
        local_cluster = Cluster(
            initialize_head=False,
            connect=False,
            head_node_args={
                "num_cpus": 0,
                "num_gpus": 0,
                "resources":{"THIS_IS_THE_HEAD_NODE":1},
            },
            shutdown_at_exit=False
        )
        #
        #
        #
        object_spill_config = json.dumps(
            {
            "type": "filesystem",
            "params": { 
                    "directory_path": [
                    "/tmp/spill_test",
                    #"/dev/shm/spill",
                    #"/dev/shm/spill_1",
                    #"/dev/shm/spill_2",
                    ]
                },
            }
        )
        test_pull_spilled_object(local_cluster, (object_spill_config,None), False)

    elif os.environ.get("PARALLEL_CI"):
        sys.exit(pytest.main(["-n", "auto", "--boxed", "-vs", __file__]))
    else:
        sys.exit(pytest.main(["-sv", __file__]))
