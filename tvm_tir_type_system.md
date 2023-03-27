
## tir的类型系统

### Var
* `Var`: 传统意义上的变量，对应一个地址
* `IterVar`: 迭代变量，包括range，迭代类型等

### Stmt
* `LetStmt`: Let binding, bind var to value, then run body.
* `AttrStmt`: 
* `AssertStmt`
* `Store`: Store value to the buffer. Equivalent to ((DType*)buffer_var)[index] = value.
* `BufferStore`: Store value to the high dimension buffer. buffer[i, j] = value;
* `BufferRealize`: Annotate the region where the buffer need to be read and write in the body. We only need to allocate the space for the corresponding region.
* `ProducerStore`: Only appear in high-level dsl
* `ProducerRealize`:类似
* `Allocate`:
* `AllocateConst`
* `DeclBuffer`: Declare a buffer that can be used in the body
* `SeqStmt`
* `IfThenElse`
* `Evaluate`: Evaluates an expression. This is mostly used for putting a Call node into Stmt. If value do not have side-effect, this node can be safely removed.
* `For`:
* `While`
* `Prefetch`
* `BufferRegion`: Representing the region of multi-dimensional buffer access.
* `MatchBufferRegion`: Match introduces a constraint that the source buffer region can be remapped to the data layout specified by the buffer field. The constraint can be checked in later part of lowering (or optionally during runtime).
* `block`: 
```C++
/*!
 * \brief A block is a basic schedule unit in TIR.
 * \note Block's body is parameterized by iter vars.
 * \code
 *
 *  with T.block(name):
 *      v0 = T.axis.S(domain, value0)
 *      v1 = T.axis.R(domain, value1)
 *      ...
 *      T.reads([buffer0[start:end, ...], ...])
 *      T.writes([buffer1[start:end, ...], ...])
 *      T.where(predicate)
 *      buffer2 = T.alloc_buffer(shape, dtype)
 *      buffer3 = T.match_buffer(source_buffer[start:end, ...])
 *      T.attr({attr_key: attr_value, ...})
 *      with T.init():
 *          // init body
 *      // body
 *
 * \endcode
 */
class BlockNode : public StmtNode {
 public:
  /*! \brief The variables of the block. */
  Array<IterVar> iter_vars;
  /*! \brief The read buffer regions of the block. */
  Array<BufferRegion> reads;
  /*! \brief The write buffer regions of the block. */
  Array<BufferRegion> writes;
  /*! \brief The name_hint of the block. */
  String name_hint;
  /*! \brief The body of the block. */
  Stmt body;
  /*!
   * \brief The init statement is executed during the first iteration of reduction loops in a
   *  reduction block. The optional init field allows us to represent initialization and
   *  reduction update in a single block and transform them collectively.
   *  We also provide primitives to decompose the init into a separate block during scheduling.
   *  Init field is `NullOpt` if there is no reduction iter_vars
   */
  Optional<Stmt> init;
  /*! \brief The buffer allocated in the block. */
  Array<Buffer> alloc_buffers;
  /*! \brief The match buffer regions. */
  Array<MatchBufferRegion> match_buffers;
  /*! \brief The annotation of the block. */
  Map<String, ObjectRef> annotations;
```

## Op
* add
* make_zero
* ceil

...

## PrimExpr
* FloatImm
* IntImm
* Cast
* BinaryOp
* Select
* BufferLoad
* Load
* Ramp
* BroadCast
* Call
* Shuffle
* CommReducer
* Reduce
* Any

## PrimFunc

## IndexMap

