# fuzz ALUs of const and mem values
from typing import List
import unittest
import numpy as np
from hypothesis import given, strategies as strat, settings
from tinygrad.codegen.uops import UOp, UOpGraph, UOps
from tinygrad.device import Buffer, Device
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv
from tinygrad.ops import BinaryOps, UnaryOps, exec_alu, python_alu
from tinygrad.renderer import Program
from tinygrad.tensor import _to_np_dtype
from test.test_dtype_alu import ht

settings.register_profile("my_profile", max_examples=500, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

def exec_uop(val:UOp, memory:List[int]) -> int:
  glbl = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (0, True))
  store = UOp(UOps.STORE, None, (glbl, UOp.const(dtypes.int, 0), val))
  uops = UOpGraph([store])
  uops.linearize()
  code = Device[Device.DEFAULT].renderer.render("test", uops)
  inputs = [Buffer(Device.DEFAULT, 1, dtypes.int, initial_value=np.array([x], _to_np_dtype(dtypes.int)).data) for x in memory]
  rawbufs = [Buffer(Device.DEFAULT, 1, dtypes.int).allocate()] + inputs
  CompiledRunner(Program("test", code, Device.DEFAULT)).exec(rawbufs)
  return np.frombuffer(rawbufs[0].as_buffer(), _to_np_dtype(dtypes.int))[0]

def load(val, memory) -> UOp:
  if val not in memory: memory.append(val)
  glbl = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (memory.index(val)+1, False))
  return UOp(UOps.LOAD, dtypes.int, (glbl, UOp.const(dtypes.int, 0)))

int_alu = [x for x in python_alu if x not in [UnaryOps.LOG2, UnaryOps.EXP2, UnaryOps.SIN, UnaryOps.RECIP, UnaryOps.SQRT]]
# TODO: with the approximations everything renders these
if not getenv("PTX"):
  int_alu.remove(BinaryOps.SHR)
  int_alu.remove(BinaryOps.SHL)

class TestFuzzUPat(unittest.TestCase):
  @given(ht.int32, ht.int32, strat.sampled_from(list(filter(lambda x:x in BinaryOps, int_alu))))
  def test_int32_binary(self, a, b, op):
    if op in [BinaryOps.SHL, BinaryOps.SHR] and b < 0: return
    if op in [BinaryOps.MOD, BinaryOps.IDIV] and b == 0: return
    # load + const
    expected = exec_alu(op, dtypes.int32, (a, b))
    memory: List[int] = []
    uop = UOp.alu(op, load(a, memory), UOp.const(dtypes.int, b))
    self.assertEqual(exec_uop(uop, memory), expected)
    # const + load
    expected = exec_alu(op, dtypes.int32, (a, b))
    memory: List[int] = []
    uop = UOp.alu(op, UOp.const(dtypes.int, a), load(b, memory))
    self.assertEqual(exec_uop(uop, memory), expected)

if __name__ == "__main__":
  unittest.main()
