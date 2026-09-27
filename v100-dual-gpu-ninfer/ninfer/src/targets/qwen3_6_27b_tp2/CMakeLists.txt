target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/variant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/tp2_comm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/tp2_mailbox.cu)

target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/export)

# TP2 collectives: an sm_70-capable NCCL (2.21 wheel build); the system NCCL has no Volta kernels.
set(NINFER_NCCL_ROOT "/home/x/linkprobe/nccl221/nvidia/nccl" CACHE PATH "NCCL with sm_70 kernels")
target_include_directories(ninfer_engine PRIVATE ${NINFER_NCCL_ROOT}/include)
target_link_libraries(ninfer_engine PUBLIC ${NINFER_NCCL_ROOT}/lib/libnccl.so.2)
set(CMAKE_BUILD_RPATH "${NINFER_NCCL_ROOT}/lib")
