/*
 * Copyright (c) Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_EFAGDA_DEVICE_H_
#define _NVSHMEMI_EFAGDA_DEVICE_H_


#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include "device/nvshmem_device_macros.h"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device_host_transport/nvshmem_common_efagda.h"
#include <stdio.h>

#include <limits.h>
#include <cuda/std/climits>
#include "device_host/nvshmem_types.h"
#include "non_abi/nvshmem_build_options.h"

#define EFAGDA_MAX_TRANSFER_SIZE (1ULL << 30)
#define NVSHMEMI_MIN(x, y) ((x) < (y) ? (x) : (y))
#define NVSHMEMI_MAX(x, y) ((x) > (y) ? (x) : (y))

#define EFA_SEND_WQE_SHIFT 6  // log2(64) since efa_io_tx_wqe is 64 bytes

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

/**
 * DO NOT use BSWAP(READ_ONCE(x)) as it could create a bug.
 * BSWAP is a pre-processor function. It will be unrolled to many READ_ONCE.
 */
#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, v) (ACCESS_ONCE(x) = (v))
#endif


// TODO REMOVE
#define __CUDA_ARCH__ 1

#ifdef __CUDA_ARCH__

#define BIT(nr)     (1UL << (nr))

#define __bf_shf(x) (__builtin_ffsll(x) - 1)

#define FIELD_GET(_mask, _reg)                                                 \
	({                                                                     \
		(typeof(_mask))(((_reg) & (_mask)) >> __bf_shf(_mask));        \
	})

#define FIELD_PREP(_mask, _val)                                                \
	({                                                                     \
		((typeof(_mask))(_val) << __bf_shf(_mask)) & (_mask);          \
	})

#define BITS_PER_LONG	   (8 * sizeof(long))

#define GENMASK(h, l) \
	(((~0UL) - (1UL << (l)) + 1) & (~0UL >> (BITS_PER_LONG - 1 - (h))))

#define EFA_GET(ptr, mask) FIELD_GET(mask##_MASK, *(ptr))

#define EFA_SET(ptr, mask, value)                                              \
	({                                                                     \
		typeof(ptr) _ptr = ptr;                                        \
		*_ptr = (*_ptr & ~(mask##_MASK)) |                             \
			FIELD_PREP(mask##_MASK, value);                        \
	})

#define container_of(ptr, type, field) \
	((type *) ((char *)ptr - offsetof(type, field)))

__device__ uint32_t efa_cq_get_current_index(const efa_cq* cq) {
    return cq->consumed_cnt & cq->queue_mask;
}

__device__ int efa_cqe_is_pending(const efa_io_cdesc_common* cqe_common, int phase) {
    volatile uint8_t *cqe_flag = (volatile uint8_t *)(&cqe_common->flags);
    return EFA_GET(cqe_flag, EFA_IO_CDESC_COMMON_PHASE) == phase;
}

__device__ efa_io_cdesc_common* efa_get_cqe(efa_cq* cq, int entry) {
    return (efa_io_cdesc_common*)(cq->buf + (entry * cq->entry_size));
}

__device__ efa_io_cdesc_common* cq_next_cqe_get(efa_cq* cq) {
    uint32_t old_consumed_cnt = cq->consumed_cnt;
    uint32_t current_index = old_consumed_cnt & cq->queue_mask;
    efa_io_cdesc_common* cqe = efa_get_cqe(cq, current_index);

    if (efa_cqe_is_pending(cqe, cq->phase)) {
        uint32_t claimed_cnt = atomicCAS(&cq->consumed_cnt, old_consumed_cnt, old_consumed_cnt + 1);
        if (claimed_cnt == old_consumed_cnt) {
            __threadfence_system();

            if (!((old_consumed_cnt + 1) & cq->queue_mask)) {
                atomicXor(&cq->phase, 1); // atomic phase flip
            }
            return cqe;
        }
    }

    return nullptr;
}

__device__ int efagda_cq_poll_next(efa_cq* cq, efa_io_cdesc_common** out_cqe) {
    efa_io_cdesc_common* cqe = cq_next_cqe_get(cq);
    if (!cqe) {
        return 0;
    }

    *out_cqe = cqe;
    return 1;
}

__device__ enum efagda_wc_opcode efagda_wc_read_opcode(efa_io_cdesc_common* cqe)
{
	enum efa_io_send_op_type op_type;

	op_type = (enum efa_io_send_op_type)EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_OP_TYPE);

	if (EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_Q_TYPE) == EFA_IO_SEND_QUEUE) {
		if (op_type == EFA_IO_RDMA_WRITE)
			return EFAGDA_WC_RDMA_WRITE;

        if (op_type == EFA_IO_RDMA_READ)
			return EFAGDA_WC_RDMA_READ;

		return EFAGDA_WC_SEND;
	}

	if (op_type == EFA_IO_RDMA_WRITE)
		return EFAGDA_WC_RECV_RDMA_WITH_IMM;

	return EFAGDA_WC_RECV;
}

__device__ bool efa_wc_is_unsolicited(efa_io_cdesc_common* cqe)
{
	return EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_UNSOLICITED);
}

__device__ uint16_t efagda_wc_read_req_id(efa_io_cdesc_common* cqe)
{
	return cqe->req_id;
}

__device__ uint32_t efagda_wc_read_vendor_err(efa_io_cdesc_common* cqe)
{
	return cqe->status;
}

__device__ bool efagda_wc_has_imm(efa_io_cdesc_common* cqe)
{
	return EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_HAS_IMM);
}

__device__ uint32_t efagda_wc_read_imm_data(efa_io_cdesc_common* cqe)
{
	struct efa_io_rx_cdesc *rcqe;

	rcqe = container_of(cqe, struct efa_io_rx_cdesc, common);

	return rcqe->imm;
}

__device__ uint32_t efagda_wc_read_byte_len(efa_io_cdesc_common* cqe)
{
	struct efa_io_cdesc_common *cqe_ptr = cqe;
	struct efa_io_rx_cdesc_ex *rcqe;
	uint32_t length;

	if (EFA_GET(&cqe_ptr->flags, EFA_IO_CDESC_COMMON_Q_TYPE) != EFA_IO_RECV_QUEUE)
		return 0;

	rcqe = container_of(cqe_ptr, struct efa_io_rx_cdesc_ex, base.common);

	length = rcqe->base.length;
	if (EFA_GET(&cqe_ptr->flags, EFA_IO_CDESC_COMMON_OP_TYPE) == EFA_IO_RDMA_WRITE)
		length |= ((uint32_t)rcqe->u.rdma_write.length_hi << 16);

	return length;
}

__device__ uint32_t efagda_wc_read_qp_num(efa_io_cdesc_common* cqe)
{
	return cqe->qp_num;
}

__device__ uint32_t efagda_wc_read_src_qp(efa_io_cdesc_common* cqe)
{
	struct efa_io_rx_cdesc *rcqe;

	rcqe = container_of(cqe, struct efa_io_rx_cdesc, common);

	return rcqe->src_qp_num;
}

__device__ uint32_t efagda_wc_read_slid(efa_io_cdesc_common* cqe)
{
	struct efa_io_rx_cdesc *rcqe;

	rcqe = container_of(cqe, struct efa_io_rx_cdesc, common);

	return rcqe->ah;
}

__device__ int efagda_sq_init_wr(struct efa_io_tx_wqe *wqe, enum efa_io_send_op_type op_type, uint16_t wr_id) {
    memset(wqe, 0, sizeof(*wqe));
    EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_META_DESC, 1);
	EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_OP_TYPE, op_type);
	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_PHASE, qp->mvars.mgmt.wqes_posted & 1);
	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_FIRST, 1);
	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_LAST, 1);
	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_COMP_REQ, 1);

    wqe->meta.req_id = wr_id;

    return 0;
}

__device__ void efagda_set_wqe_imm_data(struct efa_io_tx_wqe *wqe, uint32_t imm_data)
{
    wqe->meta.immediate_data = HTOBE32(imm_data);
	EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_HAS_IMM, 1);
}

__device__ void efagda_set_remote_mem(struct efa_io_remote_mem_addr *remote_mem, uint32_t rkey, uint64_t remote_addr)
{
	remote_mem->rkey = rkey;
	remote_mem->buf_addr_lo = remote_addr & 0xFFFFFFFF;
	remote_mem->buf_addr_hi = remote_addr >> 32;
}

__device__ void efagda_set_tx_buf(struct efa_io_tx_buf_desc *tx_buf, uint64_t addr, uint32_t lkey, uint32_t length)
{
	tx_buf->length = length;
	EFA_SET(&tx_buf->lkey, EFA_IO_TX_BUF_DESC_LKEY, lkey);
	tx_buf->buf_addr_lo = addr & 0xffffffff;
	tx_buf->buf_addr_hi = addr >> 32;
}

__device__ int efagda_init_send_wr(efa_qp* qp, uint16_t wr_id) {
    return efagda_sq_init_wr(qp, EFA_IO_SEND, wr_id);
}

__device__ int efagda_init_send_imm_wr(struct efa_io_tx_wqe *wqe, uint16_t wr_id, uint32_t imm_data)
{
    int ret;

    ret = efagda_sq_init_wr(qp, EFA_IO_SEND, wr_id);
    if (ret)
        return ret;

    efagda_set_wqe_imm_data(wqe, imm_data);

    return 0;
}

__device__ int efagda_init_rdma_read_wr(struct efa_io_tx_wqe *wqe, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr)
{
    int ret;

    ret = efagda_sq_init_wr(qp, EFA_IO_RDMA_READ, wr_id);
    if (ret)
        return ret;

	efagda_set_remote_mem(&wqe->data.rdma_req.remote_mem, rkey, remote_addr);

    return 0;
}

__device__ int efagda_init_rdma_write_wr(struct efa_io_tx_wqe *wqe, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr)
{
    int ret;

    ret = efagda_sq_init_wr(qp, EFA_IO_RDMA_WRITE, wr_id);
    if (ret)
        return ret;

	efagda_set_remote_mem(&wqe->data.rdma_req.remote_mem, rkey, remote_addr);

    return 0;
}

__device__ int efagda_init_rdma_write_imm_wr(struct efa_io_tx_wqe *wqe, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr, uint32_t imm_data)
{
    int ret;

    ret = efagda_sq_init_wr(wke, EFA_IO_RDMA_WRITE, wr_id);
    if (ret)
        return ret;

	efagda_set_remote_mem(&wqe->data.rdma_req.remote_mem, rkey, remote_addr);
    efagda_set_wqe_imm_data(wqe, imm_data);

    return 0;
}

__device__ void efagda_wr_set_remote_addr(struct efa_io_tx_wqe *wqe, uint16_t ah, uint32_t remote_qpn, uint32_t remote_qkey)
{
	wqe->meta.ah = ah;
	wqe->meta.dest_qp_num = remote_qpn;
	wqe->meta.qkey = remote_qkey;
}

__device__ int efagda_wr_set_inline_data(struct efa_io_tx_wqe *wqe, void *addr, size_t length)
{
    uint8_t op_type;

	if (length > sizeof(wqe->data.inline_data))
		return -EINVAL;

    op_type = EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_OP_TYPE);
    if (op_type != EFA_IO_SEND)
        return -EINVAL;

	EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_INLINE_MSG, 1);
	memcpy(wqe->data.inline_data, addr, length);
	wqe->meta.length = length;

    return 0;
}

__device__ int efagda_wr_set_sge(struct efa_io_tx_wqe *wqe, uint32_t lkey, uint64_t addr, uint32_t length)
{
	struct efa_io_tx_buf_desc *buf;
	struct efa_io_tx_wqe *wqe;
	uint8_t op_type;

	wqe->meta.length = 1;

	op_type = EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_OP_TYPE);
	switch (op_type) {
	case EFA_IO_SEND:
		buf = &wqe->data.sgl[0];
		break;
	case EFA_IO_RDMA_READ:
	case EFA_IO_RDMA_WRITE:
		wqe->data.rdma_req.remote_mem.length = length;
		buf = &wqe->data.rdma_req.local_mem[0];
		break;
	default:
		return -EINVAL;
	}

	efagda_set_tx_buf(buf, addr, lkey, length);
    return 0;
}

__device__ int efagda_finalize_send_wr(nvshmemi_efagda_device_qp_t *qp)
{
    atomicAdd(&qp->mvars.mgmt.wqes_posted, 1);
	atomicAdd(&qp->mvars.mgmt.prod_idx, 1);

    atomicAdd(&qp->mvars.mgmt.wqes_pending, 1);
    if (qp->mvars.mgmt.wqes_pending >= qp->mvars.attr.max_batch) {
        __threadfence_system();
        *qp->mvars.attr.db = qp->mvars.mgmt.prod_idx;
        qp->mvars.mgmt.wqes_pending = 0;
    }

    return 0;
}

__device__ void efagda_flush_send_wrs(nvshmemi_efagda_device_qp_t *qp)
{
    if (!qp->mvars.mgmt.wqes_pending)
        return;

    __threadfence_system();
    *qp->mvars.attr.db = qp->mvars.mgmt.prod_idx;
    qp->mvars.mgmt.wqes_pending = 0;
}

__device__ int efagda_post_recv_wr(efa_qp* qp, uint64_t addr, uint32_t length, uint32_t lkey) {
    struct efa_io_rx_desc wqe = {0};
    uint32_t rq_desc_offset;

    EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_FIRST, 1);
    EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_LAST, 1);

    EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_LKEY, lkey);
    wqe.buf_addr_lo = addr;
    wqe.buf_addr_hi = addr >> 32;
    wqe.length = length;

    /* Copy descriptor to RX ring */
    rq_desc_offset = (qp->rq.wq.pc & qp->rq.wq.queue_mask) * sizeof(wqe);
    memcpy(qp->rq.buf + rq_desc_offset, &wqe, sizeof(wqe));

    atomicAdd(&qp->rq.wq.pc, 1);
    if (!(qp->rq.wq.pc & qp->rq.wq.queue_mask))
        atomicAdd(&qp->rq.wq.phase, 1);

    atomicAdd(&qp->rq.wq.wqes_pending, 1);
    if (qp->rq.wq.wqes_pending == qp->rq.wq.max_batch) {
        __threadfence_system();
        *qp->rq.wq.db = qp->rq.wq.pc;

        qp->rq.wq.wqes_pending = 0;
    }

    return 0;
}

__device__ void efagda_flush_recv_wrs(efa_qp *qp)
{
    if (!qp->rq.wq.wqes_pending)
        return

    __threadfence_system();
    *qp->rq.wq.db = qp->rq.wq.pc;
    qp->rq.wq.wqes_pending = 0;
}

__device__ void efa_poll_cq(efa_cq* cq, int nwc, ibv_wc* wc, int* result)
{
    efa_io_cdesc_common* cqe;
    if (!efagda_cq_poll_next(cq, &cqe)) {
        *result = 0;
        return;
    }

    wc->wr_id = efagda_wc_read_req_id(cqe);
    wc->status = (ibv_wc_status)0;
    wc->opcode = (ibv_wc_opcode)efagda_wc_read_opcode(cqe);
    wc->vendor_err = efagda_wc_read_vendor_err(cqe);
    wc->byte_len = efagda_wc_read_byte_len(cqe);
    wc->imm_data = efagda_wc_read_imm_data(cqe);
    wc->qp_num = efagda_wc_read_qp_num(cqe);
    wc->src_qp = efagda_wc_read_src_qp(cqe);
    wc->wc_flags = 0;
    *result = 1;
}

/************************************************************************
 *  ^^^^^^^^ Generic EFA GDA Code Above, NVSHMEM Custom Code Below >>>>>>>
 ************************************************************************/

 // GOOD, DO NOT TOUCH!!!
__device__ nvshmemi_efagda_device_state_t *efagda_get_device_transport_state() {
    return (nvshmemi_efagda_device_state_t *)&nvshmemi_device_transport_state_d;
}

 // GOOD, DO NOT TOUCH!!!
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE size_t
efagda_cal_transfer_size(size_t req_size, size_t lchunk_size, size_t rchunk_size) {
    return NVSHMEMI_MIN(EFAGDA_MAX_TRANSFER_SIZE,
                        NVSHMEMI_MIN(req_size, NVSHMEMI_MIN(rchunk_size, lchunk_size)));
}

 // GOOD, DO NOT TOUCH!!!
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void efagda_get_lkey(
    uint64_t addr, uint32_t *out_lkey, size_t *out_chunk_size) {
    nvshmemi_efagda_device_state_t *state = efagda_get_device_transport_state();

    uint64_t heap_start = (uint64_t)nvshmemi_device_state_d.heap_base;
    uint64_t loffset = addr - heap_start;
    int log2_granularity = state->log2_cumem_granularity;
    uint64_t idx = loffset >> log2_granularity;

    if (idx < (nvshmemi_device_state_d.heap_size >> log2_granularity)) {
        nvshmemi_efagda_device_key_t device_key = state->lkeys[idx];
        *out_lkey = device_key.key;
        *out_chunk_size = device_key.next_addr - loffset;
        return;
    }

    printf("EFA GDA: pe=%d efagda_get_lkey out of bounds - idx=%lu, max=%lu\n",
           state->my_pe, idx, (nvshmemi_device_state_d.heap_size >> log2_granularity));
    assert(0);
}

 // GOOD, DO NOT TOUCH!!!
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void efagda_get_raddr_rkey(
    uint64_t addr, int dst_pe, uint64_t *out_raddr, uint32_t *out_rkey, size_t *out_chunk_size) {
    nvshmemi_efagda_device_state_t *state = efagda_get_device_transport_state();

    uint64_t heap_start = (uint64_t)nvshmemi_device_state_d.heap_base;
    uint64_t roffset = addr - heap_start;
    int npes = nvshmemi_device_state_d.npes;
    int log2_granularity = state->log2_cumem_granularity;

    uint64_t chunk_idx = roffset >> log2_granularity;
    uint64_t idx = chunk_idx * npes + dst_pe;
    uint64_t raddr = (uint64_t)nvshmemi_device_state_d.peer_heap_base_remote[dst_pe] + roffset;

    if (idx < (nvshmemi_device_state_d.heap_size >> log2_granularity) * npes) {
        nvshmemi_efagda_device_key_t device_key = state->rkeys[idx];
        *out_raddr = raddr;
        *out_rkey = device_key.key;
        *out_chunk_size = device_key.next_addr - roffset;
        return;
    }

    printf("EFA GDA: pe=%d efagda_get_raddr_rkey out of bounds - idx=%lu, max=%lu\n",
            state->my_pe, idx, (nvshmemi_device_state_d.heap_size >> log2_granularity) * npes);
    assert(0);

}

 // GOOD, DO NOT TOUCH!!!
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void efagda_get_ah_info(
    int dst_pe, uint16_t *out_ah, uint16_t *out_remote_qpn, uint32_t *out_remote_qkey) {
    struct cuda_ah_info ah_info;
    nvshmemi_efagda_device_state_t *state = efagda_get_device_transport_state();

    ah_info = state->cuda_ah[dst_pe];
    *out_ah = ah_info.ah;
    *out_remote_qpn = ah_info.remote_qpn;
    *out_remote_qkey = ah_info.remote_qkey;
}


__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void *efagda_get_wqe_ptr(
    nvshmemi_efagda_device_qp_t *qp, uint16_t wqe_idx) {
    uint16_t cnt = qp->tx_wq.nwqes;
    uint16_t idx = wqe_idx & (cnt - 1);
    return (void *)((uintptr_t)qp->tx_wq.wqe + (idx << EFA_SEND_WQE_SHIFT));
}


// Prevent code reordering from both compiler and GPU
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void EFAGDA_MFENCE() {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE
    asm volatile("fence.acq_rel.cta;" ::: "memory");
#else
    __threadfence_block();
#endif /* NVSHMEMI_EFAGDA_PTX_OPTIMIZATION_MFENCE */
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE int efagda_poll_cq(
    nvshmemi_efagda_device_cq_t *cq, uint64_t idx, int *error) {
    int status = 0;
    struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *)cq->cqe;

    const uint32_t ncqes = cq->ncqes;

    uint8_t opown;
    uint8_t opcode;
    uint16_t wqe_counter;
    uint16_t new_wqe_counter;

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    uint64_t start = ibgda_query_globaltimer();
    uint64_t now;
#endif

    uint64_t cons_idx = ibgda_atomic_read(cq->cons_idx);
    uint64_t new_cons_idx;

    assert(likely(cq->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ||
                  cq->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC));

    if (unlikely(cons_idx >= idx)) goto out;

#ifdef NVSHMEM_IBGDA_DEBUG
    // We can skip opcode == MLX5_CQE_INVALID check because we have already
    // initialized the CQ buffer to 0xff. With the QP depth range we enforce,
    // cons_idx cannot progress unless wqe_counter read from the CQ buffer is
    // a valid value.
    do {
        opown = ibgda_atomic_read(&cqe64->op_own);
        opcode = opown >> 4;

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        // TODO: Integrate timeout handler with the core NVSHMEM
        now = ibgda_query_globaltimer();
        status = ibgda_check_poll_timeout(cq, now, start, idx, error);
        if (status != 0) goto check_opcode;
#endif /* NVSHMEM_TIMEOUT_DEVICE_POLLING */
    } while (unlikely(opcode == MLX5_CQE_INVALID));

    // Prevent reordering of the opcode wait above
    EFAGDA_MFENCE();
#endif /* NVSHMEM_IBGDA_DEBUG */

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    start = ibgda_query_globaltimer();
#endif

    // If idx is a lot greater than cons_idx, we might get incorrect result due
    // to wqe_counter wraparound. We need to check prod_idx to be sure that idx
    // has already been submitted.
    while (unlikely(ibgda_atomic_read(cq->prod_idx) < idx))
        ;
    EFAGDA_MFENCE();

    do {
        new_wqe_counter = ibgda_atomic_read(&cqe64->wqe_counter);
        new_wqe_counter = BSWAP16(new_wqe_counter);
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        now = ibgda_query_globaltimer();
        status = ibgda_check_poll_timeout(cq, now, start, idx, error);
        if (status != 0) goto check_opcode;

        // Observe progress. Reset the timer.
        if (new_wqe_counter != wqe_counter) start = now;
#endif
        wqe_counter = new_wqe_counter;

        // Another thread may have updated cons_idx.
        cons_idx = ibgda_atomic_read(cq->cons_idx);
        if (likely(cons_idx >= idx)) goto out;
    }
    // NOTE: This while loop is part of do while above.
    // wqe_counter is the HW consumer index. However, we always maintain index
    // + 1 in SW. To be able to compare with idx, we need to use wqe_counter +
    // 1. Because wqe_counter is uint16_t, it may wraparound. Still we know for
    // sure that if idx - wqe_counter - 1 < ncqes, wqe_counter + 1 is less than
    // idx, and thus we need to wait. We don't need to wait when idx ==
    // wqe_counter + 1. That's why we use - (uint16_t)2 here to make this case
    // wraparound.
    while (unlikely(((uint16_t)((uint16_t)idx - wqe_counter - (uint16_t)2) < ncqes)));

    // new_cons_idx is uint64_t but wqe_counter is uint16_t. Thus, we get the
    // MSB from idx. We also need to take care of wraparound.
    ++wqe_counter;
    new_cons_idx =
        (idx & ~(0xffffULL) | wqe_counter) + (((uint16_t)idx > wqe_counter) ? 0x10000ULL : 0x0);
    atomicMax((unsigned long long int *)cq->cons_idx, (unsigned long long int)new_cons_idx);

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
check_opcode:
#endif

    // NVSHMEM always treats CQE errors as fatal.
    // Even if this error doesn't belong to the CQE in cons_idx,
    // we will just report and terminate the process.
    opown = ibgda_atomic_read(&cqe64->op_own);
    opcode = opown >> 4;

    if (unlikely(opcode == MLX5_CQE_REQ_ERR)) {
        ibgda_mlx5_err_cqe_t *cqe_err = (ibgda_mlx5_err_cqe_t *)cqe64;
        *error = cqe_err->syndrome;
#ifdef NVSHMEM_IBGDA_DEBUG
        __be16 wqe_counter = ibgda_atomic_read(&cqe_err->wqe_counter);
        __be32 s_wqe_opcode_qpn = ibgda_atomic_read(&cqe_err->s_wqe_opcode_qpn);
        printf(
            "[%d] got completion with err:\n"
            "   syndrome=%#x, vendor_err_synd=%#x, hw_err_synd=%#x, hw_synd_type=%#x,\n"
            "   wqe_counter=%#x, s_wqe_opcode_qpn=%#x,\n"
            "   cqn=%#x, cons_idx=%#lx, prod_idx=%#lx, idx=%#lx\n",
            nvshmemi_device_state_d.mype, cqe_err->syndrome, cqe_err->vendor_err_synd,
            cqe_err->hw_err_synd, cqe_err->hw_synd_type, BSWAP16(wqe_counter),
            BSWAP32(s_wqe_opcode_qpn), cq->cqn, cons_idx, ibgda_atomic_read(cq->prod_idx), idx);
#endif /* NVSHMEM_IBGDA_DEBUG */
        status = -1;
    }

out:
    // Prevent reordering of this function and subsequent instructions
    EFAGDA_MFENCE();

    return status;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void efagda_wait_for_slot_availability(
    nvshmemi_efagda_device_qp_t *qp, uint64_t wqe_idx) {
    int status = 0;
    int err = 0;
    uint16_t nwqes = qp->tx_wq.nwqes;

    // We don't want wqe_idx - nwqes to wraparound.
    if (likely(wqe_idx >= nwqes)) {
        nvshmemi_efagda_device_cq_t cq = *qp->tx_wq.cq;
        status = efagda_poll_cq(&cq, wqe_idx - nwqes, &err);
        // TODO: Integrate the error handler with the core NVSHMEM
        if (status) {
            printf("ibgda_poll_cq failed with error=%d.\n", err);
        }
        assert(likely(status == 0));
    }
    EFAGDA_MFENCE();
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t efagda_reserve_wqe_slots(
    nvshmemi_efagda_device_qp_t *qp, unsigned long long int num_wqes) {
    nvshmemi_efagda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t wqe_idx;

    wqe_idx = atomicAdd((unsigned long long int *)&mvars->tx_wq.resv_head, num_wqes);

    // If last slot is available, all prior slots are also available.
    efagda_wait_for_slot_availability(qp, wqe_idx + num_wqes);
    return wqe_idx;
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void efagda_lock_acquire(int *lock) {
    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0)
        while (atomicCAS(lock, 0, 1) == 1)
            ;  // Wait until we get the lock.

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD)
        EFAGDA_MFENCE();  // Prevent reordering before lock is acquired.

    // For other scopes, __syncwarp / __syncthreads guarantee the ordering
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void efagda_lock_release(int *lock) {
    // For other scopes, __syncwarp / __syncthreads guarantee the ordering
    nvshmemi_threadgroup_sync<SCOPE>();

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD)
        EFAGDA_MFENCE();  // Prevent reordering before lock is released.

    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0) ibgda_atomic_set(lock, 0);
}


__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void efagda_post_send(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t new_prod_idx) {
    nvshmemi_efagda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t old_prod_idx;

    // Update prod_idx before ringing the db so that we know which index is needed in quiet/fence.
    efagda_lock_acquire<NVSHMEMI_THREADGROUP_THREAD>(&mvars->post_send_lock);

    old_prod_idx = atomicMax((unsigned long long int *)&mvars->tx_wq.prod_idx,
                             (unsigned long long int)new_prod_idx);

    if (likely(new_prod_idx > old_prod_idx)) {
        EFAGDA_MEMBAR();
        efagda_update_dbr(qp, new_prod_idx);
        EFAGDA_MEMBAR();
        efagda_ring_db(qp, new_prod_idx);
    }

    efagda_lock_release<NVSHMEMI_THREADGROUP_THREAD>(&mvars->post_send_lock);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void efagda_submit_requests(
    nvshmemi_efagda_device_qp_t *qp, uint64_t base_wqe_idx, uint16_t num_wqes) {
    nvshmemi_efagda_device_state_t *state = efagda_get_state();
    nvshmemi_efagda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t mask = ~((uint64_t)(state->num_requests_in_batch - 1));

    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;

    unsigned long long int *ready_idx =
        (unsigned long long int *)(state->use_async_postsend ? qp->tx_wq.prod_idx
                                                             : &mvars->tx_wq.ready_head);

    EFAGDA_MEMBAR_NO_OPTIMIZATION();

    while (atomicCAS(ready_idx, (unsigned long long int)base_wqe_idx,
                     (unsigned long long int)new_wqe_idx) != base_wqe_idx)
            ;  // wait here

    EFAGDA_MFENCE();

    bool do_post_send =
        (new_wqe_idx == ACCESS_ONCE(&mvars->tx_wq.resv_head))  // No concurrent submissions
        || ((base_wqe_idx & mask) != (new_wqe_idx & mask))     // Num of not-yet-posted wqes is beyond the threshold.
        || (num_wqes >= state->num_requests_in_batch);         // The number of wqes in this submission
                                                               // reaches the threshold.

    if (do_post_send) efagda_post_send(qp, new_wqe_idx);
}



// TODO Add Warp coalesce-ing
// TODO Add need_cst
// This Assumes it is getting called with a single thread
template <nvshmemi_op_t channel_op, bool nbi>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void efagda_rma_thread(uint64_t rptr,
                                                                                uint64_t lptr,
                                                                                size_t remaining_size,
                                                                                int dst_pe) {
    if (unlikely(remaining_size == 0)) return;

    nvshmemi_efagda_device_state_t *state = efagda_get_state();
    nvshmemi_efagda_device_qp_t *qp = state->cuda_qp;
    int my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
    int tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();

    while (remaining_size > 0) {
        uint32_t lkey;
        size_t lchunk_size;
        efagda_get_lkey(lptr, &lkey, &lchunk_size);

        uint32_t rkey;
        uint64_t raddr;
        size_t rchunk_size;
        efagda_get_raddr_rkey(rptr, dst_pe, &raddr, &rkey, &rchunk_size);

        size_t transfer_size = efagda_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);


        // TODO Implement
        // TODO 1: efagda_reserve_wqe_slots
        // TODO 2: efagda_get_wqe_ptr
        // TODO 3: efagda_submit_requests


        uint64_t base_wqe_idx = efagda_reserve_wqe_slots(qp, tg_size);
        uint64_t my_wqe_idx = base_wqe_idx + my_tid;

        struct efa_io_tx_wqe *wqe = efagda_get_wqe_ptr(qp, my_wqe_idx);


        efagda_init_rdma_write_wr(wqe, 0, rkey, raddr);
        efagda_wr_set_remote_addr(wqe, ah, remote_qpn, remote_qkey);
        efagda_wr_set_sge(wqe, lkey, laddr, transfer_size);
        efagda_finalize_send_wr(wqe);


        if (my_tid == tg_size - 1) {
            // Require membar.sys to push data buffer to the point of consistency.
            if (channel_op == NVSHMEMI_OP_PUT && is_data_buf_in_sysmem) __threadfence_system();
            efagda_submit_requests(qp, base_wqe_idx, num_wqes);
        }

        remaining_size -= transfer_size;
        rptr += transfer_size;
        lptr += transfer_size;
    }

    if (!nbi && !did_quiet) {
        // CST, if required, has already been enqueued. We simply need to
        // do ibgda_quiet here.
        efagda_quiet(qp);
    }
}

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_efagda_rma_nbi(void *in_rptr, void *in_lptr,
                                                                      size_t bytes, int dst_pe) {
    if (unlikely(bytes == 0)) goto out;

    int my_tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    // Not warp 0, wait at the exit.
    if (my_tid >= tg_size) {
        goto out;
    }

    // lets define some vars
    nvshmemi_efagda_device_state_t *state = efagda_get_device_transport_state();
    int tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();

    int num_wqes;
    uint64_t base_wqe_idx;
    uint64_t my_wqe_idx;
    void *wqe_ptr;

    size_t remaining_size = bytes;
    size_t transfer_size;
    size_t my_transfer_size = 0;

    uint64_t rptr = (uint64_t)in_rptr;
    uint64_t lptr = (uint64_t)in_lptr;

    uint32_t lkey;
    uint32_t my_lkey = 0;
    uint32_t my_laddr;
    uint32_t lchunk_size;

    uint32_t rkey;
    uint32_t my_rkey = 0;
    uint32_t raddr;
    uint32_t my_raddr;
    size_t rchunk_size;

    int chunk_idx = 0;


    nvshmemi_efagda_device_qp_t *qp = state->cuda_qp; // TODO IBGDA Optimizes this, do I need to?

    // Calculate how many chunks we need to send.
    while (remaining_size > 0) {
        efagda_get_lkey(lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);
        efagda_get_raddr_rkey(rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);
        transfer_size = efagda_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);
        if (my_tid == chunk_idx) {
            my_lkey = lkey;
            my_laddr = lptr;
            my_rkey = rkey;
            my_raddr = raddr;
            my_transfer_size = transfer_size;
        }

        remaining_size -= transfer_size;
        rptr += transfer_size;
        lptr += transfer_size;

        ++chunk_idx;
    }

    // Too many chunks. Use ibgda_rma_thread to handle it instead.
    if (unlikely(chunk_idx > tg_size)) {
        if (my_tid == 0) {
            efagda_rma_thread<channel_op, nbi, support_half_av_seg>(req_rptr, req_lptr, bytes,
                                                                   dst_pe, proxy_pe);
        }
        goto out;
    }

    num_wqes = num_wqes_per_cmd * chunk_idx + (need_additional_wqe ? 1 : 0);

    if (my_tid == 0) {
        base_wqe_idx = efagda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas);
    }

    base_wqe_idx = __shfl_sync(EFAGDA_FULL_WARP, base_wqe_idx, 0);
    my_wqe_idx = base_wqe_idx + (my_tid * num_wqes_per_cmd);

    if (my_tid < chunk_idx) {
        wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
        ibgda_write_rdma_write_wqe<support_half_av_seg>(qp, dct, my_laddr, my_lkey,
                                                        my_raddr, my_rkey, my_transfer_size,
                                                        my_wqe_idx, fm_ce_se, wqe_ptrs);
    }

    nvshmemi_warp_sync();

    if (my_tid == chunk_idx - 1) {
        // Require membar.sys to push data buffer to the point of consistency.
        if (channel_op == NVSHMEMI_OP_PUT && is_data_buf_in_sysmem) __threadfence_system();

        ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes);

        if (!nbi) {
            // CST, if required, has already been enqueued. We simply need to
            // do ibgda_quiet here.
            ibgda_quiet(qp);
        }
    }

out:
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_efagda_quiet() {
    nvshmemi_threadgroup_sync<SCOPE>();
    int my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
    if (my_tid == 0) {
        nvshmemi_efagda_device_state_t *state = efagda_get_device_transport_state();
        int result;
        ibv_wc wc;

        while (state->cuda_qp->sq.wq.wqes_completed < state->cuda_qp->sq.wq.wqes_posted) {
            efa_poll_cq(state->cuda_cq, 1, &wc, &result);
            if (result > 0) {
                if (wc.vendor_err != 0) {
                    printf("[PE %d] ERROR: Thread 0 got completion, wr_id=%lu, vendor_error=%lu\n",
                            state->my_pe, (unsigned long)wc.wr_id, (unsigned long)wc.vendor_err);
                }

                atomicAdd(&state->cuda_qp->sq.wq.wqes_completed, 1);
            }
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_efagda_put_signal(
    void *rptr, void *lptr, size_t bytes, void *sig_rptr, uint64_t signal, nvshmemi_amo_t sig_op,
    int pe, bool is_nbi) {
    nvshmemi_threadgroup_sync<SCOPE>();
    int my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
    if (my_tid == 0) {
        nvshmemi_efagda_device_state_t *state = efagda_get_device_transport_state();
        struct efa_qp *qp = state->cuda_qp;

        uint32_t imm_data, new_val;
        do {
            imm_data = *state->put_signal_seq_counter;
            new_val = (imm_data + 1) & 0xFFFFFFF; // TODO Change to use #define
        } while (atomicCAS(state->put_signal_seq_counter, imm_data, new_val) != imm_data);

        uint16_t ah;
        uint16_t remote_qpn;
        uint32_t remote_qkey;
        efagda_get_ah_info(pe, &ah, &remote_qpn, &remote_qkey);

        // First do the put operation
        size_t remaining_size = bytes;
        uint64_t laddr = (uint64_t)lptr;
        uint64_t raddr_base = (uint64_t)rptr;
        int num_writes = 0;

        while (remaining_size > 0) {
            uint32_t lkey;
            size_t lchunk_size;
            efagda_get_lkey(laddr, &lkey, &lchunk_size);

            uint32_t rkey;
            uint64_t raddr;
            size_t rchunk_size;
            efagda_get_raddr_rkey(raddr_base, pe, &raddr, &rkey, &rchunk_size);

            size_t transfer_size = efagda_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);

            efagda_init_rdma_write_imm_wr(qp, 0, rkey, raddr, imm_data);
            efagda_wr_set_remote_addr(qp, ah, remote_qpn, remote_qkey);
            efagda_wr_set_sge(qp, lkey, laddr, transfer_size);
            efagda_finalize_send_wr(qp);

            laddr += transfer_size;
            raddr_base += transfer_size;
            remaining_size -= transfer_size;
            num_writes++;
        }

        uint32_t sig_rkey;
        uint64_t sig_raddr;
        size_t sig_chunk_size;
        efagda_get_raddr_rkey((uint64_t)sig_rptr, pe, &sig_raddr, &sig_rkey, &sig_chunk_size);


        // Then do the signal operation as SEND with inline data
        nvshmemt_efagda_signal_op_t signal_op;
        signal_op.type = 2; // NVSHMEMT_LIBFABRIC_MATCH
        signal_op.op = sig_op;
        signal_op.sequence_count = imm_data;
        signal_op.target_addr = (void*)sig_raddr;
        signal_op.sig_val = signal;
        signal_op.num_writes = num_writes;
        signal_op.src_pe = state->my_pe;

        // printf("[PE %d] efagda_put_signal: sending signal - type=%d, op=%d, seq=%u, target=%p, val=%lu, src_pe=%d\n",
        //        state->my_pe, signal_op.type, signal_op.op, signal_op.sequence_count, signal_op.target_addr, signal_op.sig_val, signal_op.src_pe);

        efagda_init_send_wr(qp, 0);
        efagda_wr_set_remote_addr(qp, ah, remote_qpn, remote_qkey);
        efagda_wr_set_inline_data(qp, &signal_op, sizeof(signal_op));
        efagda_finalize_send_wr(qp);

        efagda_flush_send_wrs(qp);

        // For fence()/quiet() correctness, the RX host proxy will send back
        // a write w/imm after the atomic has been applied on the RX side.
        atomicAdd(&qp->sq.wq.wqes_posted, 1);
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    if (!is_nbi) {
        printf("[PE %d] efagda_put_signal: calling quiet\n", nvshmemi_device_state_d.mype);
        nvshmemi_efagda_quiet<SCOPE>();  // TODO This is not an efficient implementation, but it should be correct.
    }
    // printf("[PE %d] efagda_put_signal: DONE\n", nvshmemi_device_state_d.mype);
}

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_efagda_rma(void *rptr, void *lptr,
                                                                 size_t bytes, int dst_pe) {
    nvshmemi_efagda_rma_nbi<SCOPE, channel_op>(rptr, lptr, bytes, dst_pe);
    nvshmemi_efagda_quiet<SCOPE>();  // TODO This is not an efficient implementation, but it should be correct.
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_efagda_enforce_consistency_at_target(
    bool use_membar) {
    printf("[PE %d] nvshmemi_efagda_enforce_consistency_at_target: Not Implemented\n", nvshmemi_device_state_d.mype);
}

#endif /* _NVSHMEMI_EFAGDA_DEVICE_H_ */
#endif /* __CUDA_ARCH__ */