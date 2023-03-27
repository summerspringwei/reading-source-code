```Python
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass
import tvm.testing
import tvm.topi.testing


def test_fuse_simple():
    """Simple testcase."""

    def before():
        x = relay.var("x", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        w = relay.squeeze(z)
        return relay.Function([x], w)

    def expected():
        x = relay.var("p", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        w = relay.squeeze(z)
        f1 = relay.Function([x], w)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        x = relay.var("x", shape=(10, 20))
        y = relay.Call(f1, [x])
        return relay.Function([x], y)

    z = before()
    zz = run_opt_pass(z, transform.FuseOps())
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)
```