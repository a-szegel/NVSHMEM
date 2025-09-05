/*
 * Copyright (c) Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_LIBFABRIC_EFAGDA_COMMON_H_
#define _NVSHMEMI_LIBFABRIC_EFAGDA_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

#define NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_SHIFT 28
#define NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_MASK ((1U << NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_SHIFT) - 1)

/**
 * The last sequence number is reserved for atomic-only operations.
 * This will not be returned by the sequence counter.
 */
#define NVSHMEM_STAGED_AMO_SEQ_NUM NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_MASK

/**
 * Type used for tracking sequence numbers for put-signal operations
 *
 * A sequence number is associated with each put-signal operation. This type
 * manages allocation of sequence numbers and return of sequence numbers via
 * acks from the remote side.
 */
struct nvshmemt_libfabric_endpoint_seq_counter_t {

    constexpr static uint32_t num_sequence_bits = NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_SHIFT;

    /**
     * Sequence counter is composed of:
     *
     * |----------<num_sequence_bits>----------|
     * | <num_category_bits> <num_index_bits>  |
     */
    constexpr static uint32_t num_category_bits = 1;
    constexpr static uint32_t num_categories = (1U << num_category_bits);

    constexpr static uint32_t num_index_bits = (num_sequence_bits - num_category_bits);

    constexpr static uint32_t index_mask = ((1U << num_index_bits) - 1);
    constexpr static uint32_t category_mask = (1U << num_index_bits);

    constexpr static uint32_t sequence_mask = NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_MASK;

    /**
     * The "category" is the bits before the index bit(s). Returns the category
     * bits shifted to the right.
     */
    constexpr __host__ __device__ static uint32_t get_category(uint32_t seq_num)
    {
        return ((seq_num & category_mask) >> num_index_bits);
    }

    /**
     * The "index" is the bits after the category bit(s)
     */
    constexpr __host__ __device__ static uint32_t get_index(uint32_t seq_num)
    {
        return seq_num & index_mask;
    }

    /* ------------------------------ */
    /* Member variables */

    uint32_t sequence_counter;
    uint32_t pending_acks[num_categories];

    /**
     * Reset counter and pending acks to zero
     */
    __host__ __device__ void reset()
    {
        sequence_counter = 0;
        memset(pending_acks, 0, sizeof(pending_acks));
    }

    /**
     * Obtain the next sequence number.
     */
    __host__ __device__ inline uint32_t next_seq_num()
    {
#ifdef __CUDA_ARCH__
        /* Note: on CUDA, the counter update needs to be atomic to account for multiple
           threads submitting a TX simultaneously. On host, this code is all locked so
           no special handling is needed. */
        return atomicInc(&sequence_counter, NVSHMEM_STAGED_AMO_SEQ_NUM);
#else
        uint32_t seq_num = sequence_counter++;
        if (sequence_counter == NVSHMEM_STAGED_AMO_SEQ_NUM) {
            sequence_counter = 0;
        }
        return seq_num;
#endif /* __CUDA_ARCH__ */
    }

    /**
     * Acquire sequence number
     */
    __host__ __device__ inline bool acquire_seq_num(uint32_t seq_num)
    {
        uint32_t category = get_category(seq_num);

        /* For first sequence number of category, check if category is available */
        if (get_index(seq_num) == 0) {
            if (pending_acks[category] != 0) {
                /* No sequence number available */
                return false;
            }
        }

#ifdef __CUDA_ARCH__
        uint32_t current = atomicAdd(&pending_acks[category], 1);
#else
        uint32_t current = pending_acks[category]++;
#endif /* __CUDA_ARCH__ */

        /* Can't have more outstanding acks than sequence numbers in the category */
        assert(current <= index_mask);

        return true;
    }

    /**
     * Mark a previously issued seq_num as complete, decremeting the pending
     * acks counter for the category
     */
    __device__ __host__ inline void return_acked_seq_num(uint32_t seq_num)
    {
        assert(seq_num != NVSHMEM_STAGED_AMO_SEQ_NUM);

        uint32_t category = get_category(seq_num);

        assert(category < num_categories);

#ifdef __CUDA_ARCH__
        uint32_t current = atomicSub(&pending_acks[category], 1);
#else
        uint32_t current = pending_acks[category]--;
#endif /* __CUDA_ARCH__ */
        assert(current > 0);
    }
};

typedef enum {
    NVSHMEMT_LIBFABRIC_IMM_PUT_SIGNAL_SEQ = 0,
    NVSHMEMT_LIBFABRIC_IMM_STAGED_ATOMIC_ACK,
} nvshmemt_libfabric_imm_cq_data_hdr_t;

#endif /* _NVSHMEMI_LIBFABRIC_EFAGDA_COMMON_H_ */
