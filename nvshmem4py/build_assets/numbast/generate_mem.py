# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

import os
import argparse
from dataclasses import dataclass

import jinja2
from numbast_common import Dtype, BaseConfig, rma_dtypes

@dataclass
class MemConfig(BaseConfig):
    """Parent Class for all Mem APIs"""
    pass

@dataclass
class GetMulticastArrayConfig(MemConfig):
    """Defines all variants of the _get_multicast_array API
    """
    dtype: list[Dtype]

@dataclass
class GetPeerArrayConfig(MemConfig):
    """Defines all variants of the _get API

    Attributes:
        src_dst: list of Dtypes
        scope: list of ExecutionScopes
        blocking_semantics: list of BlockingSemantics
    """
    dtype: list[Dtype]


def generate_get_multicast_array() -> dict:
    """Populate a jinja2 friendly dict for all variants of the _get_multicast_array API"""
    return GetMulticastArrayConfig(dtype=rma_dtypes).to_dict()

def generate_get_peer_array() -> dict:
    """Populate a jinja2 friendly dict for all variants of the _get_peer_array API"""
    return GetPeerArrayConfig(dtype=rma_dtypes).to_dict()

def generate_mem(config) -> str:
    """Load the jinja2 template and render it with the config.
    
    Config Format:
    {
        "get_multicast_array_config": {
            "src_dst": list[Dtype],
            "scope": list[ExecutionScope],
            "blocking_semantics": list[BlockingSemantics]
        },
        "get_peer_array_config": {
            "src_dst": list[Dtype],
            "scope": list[ExecutionScope],
            "blocking_semantics": list[BlockingSemantics]
        },
    }

    Returns
    -------
    str: The rendered RMA bindings
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    rma_template_path = os.path.join(this_dir, "templates", "core", "device", "numba", "mem.py.j2")
    with open(rma_template_path, "r") as f:
        template = jinja2.Template(f.read())
    return template.render(**config)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    config = {
        "get_multicast_array_config": generate_get_multicast_array(),
        "get_peer_array_config": generate_get_peer_array(),
    }
    target = os.path.join(args.output_dir, "mem.py")
    with open(target, "w") as f:
        print(f"Generating Mem bindings to {target}")
        f.write(generate_mem(config))