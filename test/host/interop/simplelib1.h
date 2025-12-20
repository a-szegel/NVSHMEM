#include <cstdio>
#include "nvshmem.h"
#include "cuda.h"
#include "cuda_runtime.h"

extern "C" {
void simplelib1_init();
int simplelib1_dowork();
void simplelib1_finalize();
}
