void sync_grids(unsigned int expected, volatile unsigned int *arrived){
  bool cta_master = (threadIdx.x + threadIdx.y + threadIdx.z == 0);
  bool gpu_master = (blockIdx.x + blockIdx.y + blockIdx.z == 0);

  // Sync all threads within a block
  __syncthreads();

  if(cta_master){ // if threadIdx = (0, 0, 0)
    unsigned int nb = 1;

    if (gpu_master){
      nb = 0x80000000 - (expected - 1);
    }
    // wait until all previous writes to shared memory and global memory are visible to other threads
    __threadfence();

    // Add arrived by one except the thread with blockIdx(0,0,0)&threadIdx(0, 0, 0)
    unsigned int oldArrive;
    oldArrive = atomicAdd(arrived, nb);

    // Thread wait until all threads add arrived to (expected-1)
    // plus gpu_master arrived become 0x80000000
    // The condition become false
    while(!((((oldArrive ^ *arrived) & 0x80000000) != 0)));

    // Let all other threads see *arrived
    unsigned int val;
    asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(val) : _CG_ASM_PTR_CONSTRAINT((unsigned int*)addr) : "memory");
    // Avoids compiler warnings from unused variable val
    (void)(val = val);
  }

  // Sync all threads within a block
  __syncthreads();
}
