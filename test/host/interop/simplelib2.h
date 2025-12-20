#include <cstdio>
#include "nvshmem.h"
#include "cuda.h"
#include "cuda_runtime.h"

extern "C" {
void simplelib2_init();
int simplelib2_dowork();
void simplelib2_finalize();
}
