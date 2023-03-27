## linalg 中算子融合源代码阅读

整个linalg的入口函数都在`include/mlir/Dialect/Linalg/Passes.h`中，对外部提供各种优化pass。关于融合的pass入口函数为`createLinalgElementwiseOpFusionPass`。

先看了`lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp`中具体融合的代码。
`getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp`这个函数就是实现的SOUFFLE的垂直融合。