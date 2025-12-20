if(GTest_FOUND)
nvshmem_add_gtest_unit_test_prefix(P2P_unit_tests.cpp "transport_" "${NVSHMEM_TEST_TLD}/../src/host/mem/mem_transport.cpp")
nvshmem_add_gtest_unit_test_prefix(remote_unit_tests.cpp "transport_" "${NVSHMEM_TEST_TLD}/../src/host/mem/mem_transport.cpp")
endif()