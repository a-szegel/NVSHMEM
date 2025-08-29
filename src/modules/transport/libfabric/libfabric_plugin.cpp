/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "libfabric.h"
#include "transport_common.h"


int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table, int api_version) {
    return nvshmemt_libfabric_init(t, table, api_version);
}
