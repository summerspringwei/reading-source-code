
# CUDA源代码解读
有关CUDA的代码在`integdev_gpu_drv\integ\gpu_drv\stage_rel\drivers\gpgpu`下，并在本文档中设置为根目录。


## cudaLaunchKernel和cudaLaunchCooperativeKernel

代码入口都在`cuda\src\api\apilaunch.c`中，其名字改变了为
`cuapiLaunchCooperativeKernel`
最终调到了`hal.launchControl`
```C++
    cuiPerformanceBegin("hal.launchControl", CUDA_PERF_GROUP_KERNEL_LAUNCH, CUDA_PERF_SUBGROUP_DEFAULT);
    ctx->device->hal.launchControl(&nvCurrent,
                                   channel,
                                   func,
                                   pTask,
                                   launchData);
    cuiPerformanceEnd("hal.launchControl", CUDA_PERF_GROUP_KERNEL_LAUNCH, CUDA_PERF_SUBGROUP_DEFAULT);
```
对应的函数签名为：
```C++
void (*launchControl)(CUnvCurrent** pnvCurrent, CUnvchannel* channel, CUfunc* func, CUtask *task, CUIlaunchData *launchData);```
<!-- 最终调到了`cuda\src\api\apilaunch.c:296`的`cuiLaunch` -->
