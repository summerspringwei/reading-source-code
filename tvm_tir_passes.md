## 解读TVM在codegen过程中的各种pass

### LowerMatchBuffer
其前序pass是`CompactBufferAllocation`，计算cuda一个block内的需要分配的Buffer是多大。
`CompactBufferAllocation`的intrinsic输入是：

```C++
block([], "relu_a_shared_o") {
tir.reads([relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.  y*2):((threadIdx.y*2) + 2)]])
tir.writes([relu_a_shared[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8),      (threadIdx.y*2):((threadIdx.y*2) + 2)]])
shared_buff = match_buffer(relu_a_shared[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx. x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)])
global_buff = match_buffer(relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) +  8), (threadIdx.y*2):((threadIdx.y*2) + 2)])
@tir.oraa_slice_tensor(2, 2, 8, 2, shared_buff_1: Pointer(shared int8), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), global_buff_1:  Pointer(global int8), elem_offset: int32, (global_buff_s0: int32*2), 1, dtype=handle), elem_offset, global_buff_s0, global_buff_s1: int32, global_buff_s2:        int32, "int8", dtype=handle)
```
经过`CompactBufferAllocation`后：
```Rust
block([], "relu_a_shared_o") {
tir.reads([relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.  y*2):((threadIdx.y*2) + 2)]])
tir.writes([relu_a_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) - (blockIdx.y*2)):         (((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
shared_buff = match_buffer(relu_a_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) -           (blockIdx.y*2)):(((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)])
global_buff = match_buffer(relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) +  8), (threadIdx.y*2):((threadIdx.y*2) + 2)])
@tir.oraa_slice_tensor(2, 2, 8, 2, shared_buff_1: Pointer(shared int8), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), global_buff_1:  Pointer(global int8), elem_offset: int32, (global_buff_s0: int32*2), 1, dtype=handle), elem_offset, global_buff_s0, global_buff_s1: int32, global_buff_s2:        int32, "int8", dtype=handle)
```
经过`LowerMatchBuffer`的intrinsic后：
```Rust
block([], "relu_a_shared_o") {
tir.reads([relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.  y*2):((threadIdx.y*2) + 2)]])
tir.writes([relu_a_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) - (blockIdx.y*2)):         (((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
@tir.oraa_slice_tensor(2, 2, 8, 2, relu_a_shared_1: Pointer(shared int8), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), relu_a_1,     ((((blockIdx.x*2097152) + (blockIdx.y*8192)) + (threadIdx.x*512)) + (threadIdx.y*2)), (1048576*2), 1, dtype=handle), ((((blockIdx.x*2097152) + (blockIdx.y*       8192)) + (threadIdx.x*512)) + (threadIdx.y*2)), 1048576, 4096, 64, "int8", dtype=handle)
```