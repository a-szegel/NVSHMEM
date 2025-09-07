#pragma once

/* #undef NVSHMEM_COMPLEX_SUPPORT */
/* #undef NVSHMEM_DEBUG */
/* #undef NVSHMEM_DEVEL */
/* #undef NVSHMEM_TRACE */
/* #undef NVSHMEM_DEFAULT_PMI2 */
/* #undef NVSHMEM_DEFAULT_PMIX */
/* #undef NVSHMEM_DEFAULT_UCX */
/* #undef NVSHMEM_GPU_COLL_USE_LDST */
/* #undef NVSHMEM_IBDEVX_SUPPORT */
#define NVSHMEM_IBRC_SUPPORT
#define NVSHMEM_LIBFABRIC_SUPPORT
#define NVSHMEM_EFAGDA_SUPPORT
#define NVSHMEM_MPI_SUPPORT
#define NVSHMEM_NVTX
/* #undef NVSHMEM_PMIX_SUPPORT */
/* #undef NVSHMEM_SHMEM_SUPPORT */
/* #undef NVSHMEM_TEST_STATIC_LIB */
/* #undef NVSHMEM_TIMEOUT_DEVICE_POLLING */
/* #undef NVSHMEM_UCX_SUPPORT */
/* #undef NVSHMEM_USE_DLMALLOC */
/* #undef NVSHMEM_USE_NCCL */
#define NVSHMEM_USE_GDRCOPY
/* #undef NVSHMEM_VERBOSE */
#define NVSHMEM_BUILD_TESTS
#define NVSHMEM_BUILD_EXAMPLES
/* #undef NVSHMEM_IBGDA_SUPPORT */
/* #undef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY */
/* #undef NVSHMEM_ENABLE_ALL_DEVICE_INLINING */
/* #undef NVSHMEM_HOSTLIB_ONLY */

/* TODO: When building the bitcode library, we need to check if EFA can be used.
 * This check for IBGDA will likely be removed in ~3.5.
 */
#if defined __clang_llvm_bitcode_lib__ || defined NVSHMEM_HOSTLIB_ONLY
#undef NVSHMEM_IBGDA_SUPPORT
#undef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
#define NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#endif
