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

如何被绑定数值的
`element_offset`被绑定到`((((blockIdx.x: int32*2097152) + (blockIdx.y: int32*8192)) + (threadIdx.x: int32*512)) + (threadIdx.y: int32*2))`
```C++
    at /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:253                                                              │·················································
#1  0x00007fffbfe6792d in tvm::tir::MatchBufferLower::CheckAndUpdateVarMap (this=0x7fffffffad90, match_buffer=...)                                   │·················································
    at /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:201                                                              │·················································
#2  0x00007fffbfe64cc1 in tvm::tir::MatchBufferLower::VisitStmt_ (this=0x7fffffffad90, op=0x555556a64be0)                                            │·················································
    at /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:49
```



```C++
#0  tvm::tir::MatchBufferLower::Bind (this=0x7fffffffad90, arg=..., value=..., arg_name="global_buff.strides_3")                                     │·················································
    at /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:253                                                              │·················································
#1  0x00007fffbfe680e5 in tvm::tir::MatchBufferLower::CheckAndUpdateVarMap (this=0x7fffffffad90, match_buffer=...)                                   │·················································
    at /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:223                                                              │·················································
#2  0x00007fffbfe64cc1 in tvm::tir::MatchBufferLower::VisitStmt_ (this=0x7fffffffad90, op=0x555556a64be0)                                            │·················································
    at /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:49
```

关于bind，其作用就是，首先，把value根据var_map进行替换，尽可能把variable替换成具体的PrimExpr；
只对，对于arg，如果arg是`Var`类型并且`var_map`里面没有，则把arg和和value绑定，
否则，则要判断arg和value是不是一样的PrimExpr，如果不是，则报错。
```C++
void Bind(const PrimExpr& arg, PrimExpr value, const std::string& arg_name = "argument") {
    CHECK_EQ(arg.dtype(), value.dtype())
        << "The data type mismatched: " << arg->dtype << " vs. " << value->dtype;
    // Handle recursive case
    value = Substitute(std::move(value), var_map_);
    if (arg->IsInstance<VarNode>()) {
      Var v = Downcast<Var>(arg);
      auto it = var_map_.find(v);
      if (it == var_map_.end()) {
        VLOG(2) <<"Bind " << v << " " << value;
        var_map_.Set(v, value);
        analyzer_.Bind(v, value);
      } else {
        AssertBinding((*it).second, value, arg_name);
      }
    } else {
      AssertBinding(arg, value, arg_name);
    }
  }
```


```Rust

[15:45:09] /home/xiachunwei/Projects/intel_tvm/src/ir/transform.cc:455: The meta data of the pass - pass name: tir.CompactBufferAllocation, opt_level: 0, required passes: []
// 在function的buffers中： relu_a->relu_a_1
// 在intrisic中，是relu_a_1
// 在match_buffer中，是从global_buff->relu_a
@main = primfn(var_relu_a: handle, var_compute: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {relu_a: Buffer(relu_a_1: Pointer(global int8), int8, [256, 256, 64, 64], []),
             compute: Buffer(compute_1: Pointer(global int8), int8, [256, 256, 64, 64], [])}
  buffer_map = {var_relu_a: relu_a, var_compute: compute} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (blockIdx.x: int32, 0, 128) "thread_binding" {
      for (blockIdx.y: int32, 0, 128) "thread_binding" {
        for (threadIdx.x: int32, 0, 8) "thread_binding" {
          for (threadIdx.y: int32, 0, 32) "thread_binding" {
            block([], "") {
              tir.reads([relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
              tir.writes([compute[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
              relu_a_shared = alloc_buffer(int8[2, 2, 64, 64])
              compute_shared = alloc_buffer(int8[2, 2, 64, 64])
               {
                block([], "relu_a_shared_o") {
                  tir.reads([relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
                  tir.writes([relu_a_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) - (blockIdx.y*2)):(((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
                  shared_buff = match_buffer(relu_a_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) - (blockIdx.y*2)):(((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)])
                  global_buff = match_buffer(relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)])
                  @tir.oraa_slice_tensor(2, 2, 8, 2, shared_buff_1: Pointer(shared int8), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), global_buff_1: Pointer(global int8), elem_offset: int32, (global_buff_s0: int32*2), 1, dtype=handle), elem_offset, global_buff_s0, global_buff_s1: int32, global_buff_s2: int32, "int8", dtype=handle)
                block([], "compute_o") {
                  tir.reads([relu_a_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) - (blockIdx.y*2)):(((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
                  tir.writes([compute_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) - (blockIdx.y*2)):(((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
                  for (n_1: int32, 0, 2) {
                    for (c_1: int32, 0, 2) {
                      for (h_1: int32, 0, 8) {
                        for (w_1: int32, 0, 2) {
                          block([], "compute") {
                            tir.reads([relu_a_shared[(((blockIdx.x*2) + n_1) - (blockIdx.x*2)), (((blockIdx.y*2) + c_1) - (blockIdx.y*2)), ((threadIdx.x*8) + h_1), ((threadIdx.y*2) + w_1)]])
                            tir.writes([compute_shared[(((blockIdx.x*2) + n_1) - (blockIdx.x*2)), (((blockIdx.y*2) + c_1) - (blockIdx.y*2)), ((threadIdx.x*8) + h_1), ((threadIdx.y*2) + w_1)]])
                            compute_shared[(((blockIdx.x*2) + n_1) - (blockIdx.x*2)), (((blockIdx.y*2) + c_1) - (blockIdx.y*2)), ((threadIdx.x*8) + h_1), ((threadIdx.y*2) + w_1)] = max(relu_a_shared[(((blockIdx.x*2) + n_1) - (blockIdx.x*2)), (((blockIdx.y*2) + c_1) - (blockIdx.y*2)), ((threadIdx.x*8) + h_1), ((threadIdx.y*2) + w_1)], 0i8)
                        }
                      }
                    }
                  }
                for (ax0: int32, 0, 2) {
                  for (ax1: int32, 0, 2) {
                    for (ax2: int32, 0, 8) {
                      for (ax3: int32, 0, 2) {
                        block([], "compute_shared") {
                          tir.reads([compute_shared[(((blockIdx.x*2) + ax0) - (blockIdx.x*2)), (((blockIdx.y*2) + ax1) - (blockIdx.y*2)), ((threadIdx.x*8) + ax2), ((threadIdx.y*2) + ax3)]])
                          tir.writes([compute[((blockIdx.x*2) + ax0), ((blockIdx.y*2) + ax1), ((threadIdx.x*8) + ax2), ((threadIdx.y*2) + ax3)]])
                          compute[((blockIdx.x*2) + ax0), ((blockIdx.y*2) + ax1), ((threadIdx.x*8) + ax2), ((threadIdx.y*2) + ax3)] = compute_shared[(((blockIdx.x*2) + ax0) - (blockIdx.x*2)), (((blockIdx.y*2) + ax1) - (blockIdx.y*2)), ((threadIdx.x*8) + ax2), ((threadIdx.y*2) + ax3)]
                      }
                    }
                  }
                }
              }
          }
        }
      }
    }
}


[15:45:09] /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:48: shared_buff = match_buffer(relu_a_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) - (blockIdx.y*2)):(((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)])


[15:45:09] /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:153: relu_a_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) - (blockIdx.y*2)):(((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]
[15:45:09] /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:190: [((threadIdx.x*512) + (threadIdx.y*2))]
[15:45:09] /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:191: elem_offset: int32
[15:45:09] /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:48: global_buff = match_buffer(relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)])


[15:45:09] /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:153: relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]
[15:45:09] /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:190: [((((blockIdx.x*2097152) + (blockIdx.y*8192)) + (threadIdx.x*512)) + (threadIdx.y*2))]
[15:45:09] /home/xiachunwei/Projects/intel_tvm/src/tir/transforms/lower_match_buffer.cc:191: elem_offset: int32
[15:45:09] /home/xiachunwei/Projects/intel_tvm/src/ir/transform.cc:455: The meta data of the pass - pass name: tir.LowerMatchBuffer, opt_level: 0, required passes: []

// 在function的buffers中： relu_a->relu_a_1
// 在intrisic中，是relu_a_1
// 在match_buffer中，是从global_buff->relu_a
@main = primfn(var_relu_a: handle, var_compute: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {relu_a: Buffer(relu_a_1: Pointer(global int8), int8, [256, 256, 64, 64], []),
             compute: Buffer(compute_1: Pointer(global int8), int8, [256, 256, 64, 64], [])}
  buffer_map = {var_relu_a: relu_a, var_compute: compute} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (blockIdx.x: int32, 0, 128) "thread_binding" {
      for (blockIdx.y: int32, 0, 128) "thread_binding" {
        for (threadIdx.x: int32, 0, 8) "thread_binding" {
          for (threadIdx.y: int32, 0, 32) "thread_binding" {
            block([], "") {
              tir.reads([relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
              tir.writes([compute[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
              relu_a_shared = alloc_buffer(int8[2, 2, 64, 64])
              compute_shared = alloc_buffer(int8[2, 2, 64, 64])
               {
                block([], "relu_a_shared_o") {
                  tir.reads([relu_a[(blockIdx.x*2):((blockIdx.x*2) + 2), (blockIdx.y*2):((blockIdx.y*2) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
                  tir.writes([relu_a_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) - (blockIdx.y*2)):(((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
                  @tir.oraa_slice_tensor(2, 2, 8, 2, relu_a_shared_1: Pointer(shared int8), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), relu_a_1, ((((blockIdx.x*2097152) + (blockIdx.y*8192)) + (threadIdx.x*512)) + (threadIdx.y*2)), (1048576*2), 1, dtype=handle), ((((blockIdx.x*2097152) + (blockIdx.y*8192)) + (threadIdx.x*512)) + (threadIdx.y*2)), 1048576, 4096, 64, "int8", dtype=handle)
                block([], "compute_o") {
                  tir.reads([relu_a_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) - (blockIdx.y*2)):(((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
                  tir.writes([compute_shared[((blockIdx.x*2) - (blockIdx.x*2)):(((blockIdx.x*2) - (blockIdx.x*2)) + 2), ((blockIdx.y*2) - (blockIdx.y*2)):(((blockIdx.y*2) - (blockIdx.y*2)) + 2), (threadIdx.x*8):((threadIdx.x*8) + 8), (threadIdx.y*2):((threadIdx.y*2) + 2)]])
                  for (n_1: int32, 0, 2) {
                    for (c_1: int32, 0, 2) {
                      for (h_1: int32, 0, 8) {
                        for (w_1: int32, 0, 2) {
                          block([], "compute") {
                            tir.reads([relu_a_shared[(((blockIdx.x*2) + n_1) - (blockIdx.x*2)), (((blockIdx.y*2) + c_1) - (blockIdx.y*2)), ((threadIdx.x*8) + h_1), ((threadIdx.y*2) + w_1)]])
                            tir.writes([compute_shared[(((blockIdx.x*2) + n_1) - (blockIdx.x*2)), (((blockIdx.y*2) + c_1) - (blockIdx.y*2)), ((threadIdx.x*8) + h_1), ((threadIdx.y*2) + w_1)]])
                            compute_shared[(((blockIdx.x*2) + n_1) - (blockIdx.x*2)), (((blockIdx.y*2) + c_1) - (blockIdx.y*2)), ((threadIdx.x*8) + h_1), ((threadIdx.y*2) + w_1)] = max(relu_a_shared[(((blockIdx.x*2) + n_1) - (blockIdx.x*2)), (((blockIdx.y*2) + c_1) - (blockIdx.y*2)), ((threadIdx.x*8) + h_1), ((threadIdx.y*2) + w_1)], 0i8)
                        }
                      }
                    }
                  }
                for (ax0: int32, 0, 2) {
                  for (ax1: int32, 0, 2) {
                    for (ax2: int32, 0, 8) {
                      for (ax3: int32, 0, 2) {
                        block([], "compute_shared") {
                          tir.reads([compute_shared[(((blockIdx.x*2) + ax0) - (blockIdx.x*2)), (((blockIdx.y*2) + ax1) - (blockIdx.y*2)), ((threadIdx.x*8) + ax2), ((threadIdx.y*2) + ax3)]])
                          tir.writes([compute[((blockIdx.x*2) + ax0), ((blockIdx.y*2) + ax1), ((threadIdx.x*8) + ax2), ((threadIdx.y*2) + ax3)]])
                          compute[((blockIdx.x*2) + ax0), ((blockIdx.y*2) + ax1), ((threadIdx.x*8) + ax2), ((threadIdx.y*2) + ax3)] = compute_shared[(((blockIdx.x*2) + ax0) - (blockIdx.x*2)), (((blockIdx.y*2) + ax1) - (blockIdx.y*2)), ((threadIdx.x*8) + ax2), ((threadIdx.y*2) + ax3)]
                      }
                    }
                  }
                }
              }
          }
        }
      }
    }
}

@tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), 
  relu_a_1,     
  ((((blockIdx.x*2097152) + (blockIdx.y*8192)) + (threadIdx.x*512)) + (threadIdx.y*2)), 
  (1048576*2), 
  1, 
  dtype=handle)
```