#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import DATASET_REGISTRY, build_dataset  # noqa
from .thumos import Thumos # noqa
from .tvseries import Tvseries  # noqa
from .hdd import Hdd  # noqa

try:
    from .ptv_datasets import Ptvcharades, Ptvkinetics, Ptvssv2  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
