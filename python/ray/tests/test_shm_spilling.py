import json
import platform
import re
import sys
import zlib

import numpy as np
import pytest

import ray
from ray._private.test_utils import wait_for_condition
from ray.cluster_utils import Cluster, cluster_not_supported
from ray.tests.test_object_spilling import assert_no_thrashing, is_dir_empty

# Note: Disk write speed can be as low as 6 MiB/s in AWS Mac instances, so we have to
# increase the timeout.

# have to figure out which node spills what and which pulls 

def test_pull_spilled_object(
    ray_start_cluster_enabled, multi_node_object_spilling_config, shutdown_only
):
    cluster = ray_start_cluster_enabled
    object_spilling_config, _ = multi_node_object_spilling_config

    # Head node.
    cluster.add_node(
        num_cpus=1,
        resources={"pool_id_0": 1, 'a':1},
        object_store_memory= 75 * 1024 * 1024, # 75 * 1024 * 1024
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
        num_cpus=1, resources={"pool_id_0": 1, 'b':1}, object_store_memory=75 * 1024 * 1024
    )
    cluster.wait_for_nodes()

    # create the objects on the remote node?
    @ray.remote(num_cpus=1, resources={"b": 1})
    def create_objects():
        results = []
        # TODO(maxwell) less than three here will mess up the program???
        # ray.get below here does nothing at all
        data_to_spill = [10 for _ in range(5)]
        for size in data_to_spill:
            arr = np.random.rand(size * 1024 * 1024)
            hash_value = zlib.crc32(arr.tobytes())
            results.append([ray.put(arr), hash_value])
        
        # ensure the objects are spilled
        # np.random.rand(5 * 1024 * 1024) this puts about 41.943152 MB in storage
        arr = np.random.rand(5 * 1024 * 1024) # SHOULD SEE EXACTLY 1 RESTORE FROM POOL ON RAYLET 1
        ray.get(ray.put(arr))
        ray.get(ray.put(arr))
        return results

    # get the objects and place into head node?
    @ray.remote(num_cpus=1, resources={"a": 1})
    def get_object(arr):
        return zlib.crc32(arr.tobytes())

    # create the objects on the remote node?
    results = ray.get(create_objects.remote())

    # restore on remote node and send over to head node?
    for value_ref, hash_value in results:
        hash_value1 = ray.get(get_object.remote(value_ref))
        assert hash_value == hash_value1

    # create the objects on the remote node?
    #results2 = ray.get(create_objects.remote())
    #for value_ref, hash_value in results2:
    #    hash_value1 = ray.get(get_object.remote(value_ref))
    #    assert hash_value == hash_value1

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
                    "/dev/shm/spill_test"
                    #"/tmp/spill_test",
                    #"/dev/shm/spill",
                    #"/dev/shm/spill_1",
                    #"/dev/shm/spill_2",
                    ]
                },
            }
        )
        test_pull_spilled_object(local_cluster, (object_spill_config,None), False)
