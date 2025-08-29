My environment at the time of developing this patch:


# ENV VARS FOR NVSHMEM Project

export PF_HOME=$HOME/PortaFiducia
export MPI_HOME=/opt/amazon/openmpi/
export LIBFABRIC_HOME=$HOME/libfabric/install
export NVSHMEM_HOME=$HOME/Nvshmem_install/
export NVSHMEM_PREFIX=$NVSHMEM_HOME
export NVSHMEM_PERFTEST_INSTALL=$NVSHMEM_HOME/bin/perftest/
export CUDA_HOME=/usr/local/cuda/
export NVSHMEM_REMOTE_TRANSPORT=efagda
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export GDRCOPY_HOME=/usr/local/gdrdrv
export NVSHMEM_MPI_SUPPORT=1 
export NVSHMEM_LIBFABRIC_SUPPORT=1 
export FI_LOG_LEVEL=warn
#export NVSHMEM_BOOTSTRAP_TWO_STAGE=true

# COMPILE NVSHMEM IN DEBUG!
export NVSHMEM_DEBUG=1 

# SET LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$MPI_HOME/lib:$CUDA_HOME/lib64:$LIBFABRIC_HOME/lib

export FI_EFA_ENABLE_SHM_TRANSFER=0


# TODO TODO
# TEMP USE Jessie's rdma-core for GDA testing
export LD_LIBRARY_PATH=/home/jiaxiyan/PortaFiducia/build/libraries/rdma_core/pr1608/rdma-core/build/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=/home/jiaxiyan/PortaFiducia/build/libraries/rdma_core/pr1608/rdma-core/build/lib:${LIBRARY_PATH}
export CPPFLAGS="-I/home/jiaxiyan/PortaFiducia/build/libraries/rdma_core/pr1608/rdma-core/build/include $CPPFLAGS"
export LDFLAGS="-L/home/jiaxiyan/PortaFiducia/build/libraries/rdma_core/pr1608/rdma-core/build/lib $LDFLAGS"


# NO NCCL
#export NVSHMEM_DEBUG=warn
#export NCCL_HOME=$PF_HOME/build/libraries/nccl/v2.26.2-1/install
#export NVSHMEM_USE_NCCL=1 
#export NCCL_DEBUG=WARN
#export LD_LIBRARY_PATH=$NCCL_HOME/lib/:$NVSHMEM_HOME/lib:$MPI_HOME/lib:$CUDA_HOME/lib64:$LIBFABRIC_HOME/lib

# Libfabric Only LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=${LIBFABRIC_HOME}:${LD_LIBRARY_PATH}


# LTTNG SUPPORT
# export LTTNG_HOME=/usr
# export NVSHMEM_LTTNG_SUPPORT=ON
# export LTTNG_HOME=$HOME

alias nvshmem_run_host_bw="$MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node  --hostfile $HOME/PortaFiducia/hostfile  -x NVSHMEMTEST_USE_MPI_LAUNCHER=1 -x LD_LIBRARY_PATH -x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_LIBFABRIC_PROVIDER -x NVSHMEM_DEBUG -x FI_LOG_LEVEL -x NVSHMEM_BOOTSTRAP_TWO_STAGE $NVSHMEM_PERFTEST_INSTALL/host/pt-to-pt/bw"

alias nvshmem_run_col="$MPI_HOME/bin/mpirun -n 16 --map-by ppr:8:node  --hostfile $HOME/PortaFiducia/hostfile  -x NVSHMEMTEST_USE_MPI_LAUNCHER=1 -x LD_LIBRARY_PATH -x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_LIBFABRIC_PROVIDER -x NVSHMEM_DEBUG -x FI_LOG_LEVEL -x NVSHMEM_BOOTSTRAP_TWO_STAGE $NVSHMEM_PERFTEST_INSTALL/device/coll/alltoall_latency"

alias nvshmem_run_put_pingpong="$MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node  --hostfile /home/szegel/hostfile  -x NVSHMEMTEST_USE_MPI_LAUNCHER=1 -x LD_LIBRARY_PATH -x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_LIBFABRIC_PROVIDER -x NVSHMEM_DEBUG -x FI_LOG_LEVEL /home/szegel/Nvshmem_install/bin/perftest/device/pt-to-pt/shmem_put_ping_pong_latency"

alias nvshmem_run_put_with_signal="$MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node  --hostfile /home/szegel/hostfile  -x NVSHMEMTEST_USE_MPI_LAUNCHER=1 -x LD_LIBRARY_PATH -x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_LIBFABRIC_PROVIDER -x NVSHMEM_DEBUG -x FI_LOG_LEVEL -x NVSHMEM_BOOTSTRAP_TWO_STAGE /home/szegel/Nvshmem_install/bin/perftest/device/pt-to-pt/shmem_put_signal_ping_pong_latency"


# EXPECTED TO FAIL
alias nvshmem_run_atomic_pingpong_segfault="$MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node  --hostfile /home/szegel/hostfile  -x NVSHMEMTEST_USE_MPI_LAUNCHER=1 -x LD_LIBRARY_PATH -x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_LIBFABRIC_PROVIDER -x NVSHMEM_DEBUG -x FI_LOG_LEVEL -x NVSHMEM_BOOTSTRAP_TWO_STAGE /home/szegel/Nvshmem_install/bin/perftest/device/pt-to-pt/shmem_atomic_ping_pong_latency"

alias nvshmrun='${HYDRA_INSTALL}/bin/nvshmrun'

alias cmake_nvshmem='cmake .. -DNVSHMEM_EFAGDA_SUPPORT=ON -DNVSHMEM_BUILD_PYTHON_LIB=0'

# ENV VARS FOiR NVSHMEM Project
