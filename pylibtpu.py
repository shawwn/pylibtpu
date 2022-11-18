import sys
import ctypes
import dataclasses
import json
from pprint import pprint as pp

from google.protobuf.json_format import MessageToJson
import os

sys.path.append(os.path.dirname(__file__))

import tpu_driver_pb2


libtpu = ctypes.cdll.LoadLibrary("libtpu.so")

FuncPtr = ctypes.CFUNCTYPE(None)

class TpuDriver(ctypes.Structure):
  pass

TpuDriver_p = ctypes.POINTER(TpuDriver)

class TpuEvent(ctypes.Structure):
  pass

TpuEvent_p = ctypes.POINTER(TpuEvent)
TpuEventList = ctypes.POINTER(TpuEvent_p)

class TpuCompiledProgramHandle(ctypes.Structure):
  _fields_ = [
      ("internal_handle", ctypes.c_void_p),
      ("event", TpuEvent_p),
      ]

TpuCompiledProgramHandle_p = ctypes.POINTER(TpuCompiledProgramHandle)

class TpuSystemInfo(ctypes.Structure):
  _fields_ = [
      ("bytes", ctypes.POINTER(ctypes.c_char)),
      ("size", ctypes.c_int32),
      ]
  def __bytes__(self):
    if self.bytes:
      return self.bytes[0:self.size]

TpuSystemInfo_p = ctypes.POINTER(TpuSystemInfo)


@dataclasses.dataclass
class TpuStatus(ctypes.Structure):
  code: int
  msg: str
  _fields_ = [
      ("code", ctypes.c_int32),
      ("msg", ctypes.c_char_p),
      ]

TpuStatus_p = ctypes.POINTER(TpuStatus)


@dataclasses.dataclass
class TpuDriverFn(ctypes.Structure):
  _fields_ = [
    ("TpuDriver_Open", ctypes.CFUNCTYPE(TpuDriver_p, ctypes.c_char_p)),
    ("TpuDriver_Close", ctypes.CFUNCTYPE(None, TpuDriver_p)),
    ("TpuDriver_Reset", ctypes.CFUNCTYPE(TpuStatus_p, TpuDriver_p)),
    ("TpuDriver_ComputeLinearizedBytesFromShape", FuncPtr),
    ("TpuDriver_QuerySystemInfo", ctypes.CFUNCTYPE(TpuSystemInfo_p, TpuDriver_p)),
    ("TpuDriver_FreeSystemInfo", ctypes.CFUNCTYPE(None, TpuSystemInfo_p)),
    ("TpuDriver_LinearizeShape", FuncPtr),
    ("TpuDriver_DelinearizeShape", FuncPtr),
    ("TpuDriver_CompileProgram", FuncPtr),
    ("TpuDriver_CompileProgramFromText", ctypes.CFUNCTYPE(TpuCompiledProgramHandle_p, TpuDriver_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(TpuEvent_p))),
    ("TpuDriver_FreeCompiledProgramHandle", FuncPtr),
    ("TpuDriver_LoadProgram", ctypes.CFUNCTYPE(ctypes.c_void_p, TpuDriver_p, ctypes.c_int32, TpuCompiledProgramHandle_p, ctypes.c_int32, ctypes.POINTER(TpuEvent_p))),
    ("TpuDriver_UnloadProgram", FuncPtr),
    ("TpuDriver_ExecuteProgram", FuncPtr),
    ("TpuDriver_AllocateTuple", FuncPtr),
    ("TpuDriver_Allocate", FuncPtr),
    ("TpuDriver_AllocateShape", FuncPtr),
    ("TpuDriver_Deallocate", FuncPtr),
    ("TpuDriver_TransferToDevice", FuncPtr),
    ("TpuDriver_TransferFromDevice", FuncPtr),
    ("TpuDriver_TransferFromDeviceToDevice", FuncPtr),
    ("TpuDriver_GetCompiledProgramShape", FuncPtr),
    ("TpuDriver_FreeCompiledProgramShape", FuncPtr),
    ("TpuDriver_EventAddCallback", FuncPtr),
    ("TpuDriver_EventAwait", FuncPtr),
    ("TpuDriver_FreeEvent", FuncPtr),
    ("TpuDriver_FreeStatus", FuncPtr),
    ("TpuDriver_Version", ctypes.CFUNCTYPE(ctypes.c_char_p)),
  ]
  @property
  def version(self) -> str:
    if self.TpuDriver_Version:
      return self.TpuDriver_Version().decode('utf8')
  def __repr__(self):
    return f"{type(self).__name__}(version={self.version!r})"



libtpu.TpuDriver_Initialize.argtypes = [ctypes.POINTER(TpuDriverFn), ctypes.c_bool]
libtpu.TpuDriver_Initialize.restype = None


driver_fn = TpuDriverFn()


libtpu.TpuDriver_Initialize(driver_fn, True)

print(driver_fn.TpuDriver_Version())
print(driver_fn)


print("opening local://")
driver = driver_fn.TpuDriver_Open(b"local://")
print(bool(driver))

# print("------ Resetting ------\n")
# status_p = driver_fn.TpuDriver_Reset(driver)
# if status_p:
#   print(status_p.contents)

print("------ Going to Query for System Information ------\n")
info_p = driver_fn.TpuDriver_QuerySystemInfo(driver)
if not info_p:
  print("failed!")
else:
  info = bytes(info_p.contents)
  info_p = driver_fn.TpuDriver_FreeSystemInfo(info_p)
  print("System Information:", info)

  system_info = tpu_driver_pb2.SystemInfo.FromString(info)
  system_info_json = json.loads(MessageToJson(system_info))
  pp(system_info_json)


hlo_module_reduce_text = b"""
HloModule BadReduce
Sum {
  x.1 = f32[] parameter(0)
  y.1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(x.1, y.1)
}
ENTRY reduce.1 {
  parameter = f32[2,2,2,3]{3,2,1,0} parameter(0)
  init_value = f32[] constant(0)
  reduce = f32[2,2,3]{2,1,0} reduce(parameter, init_value), dimensions={1}, to_apply=Sum
  ROOT copy = f32[2,2,3]{2,1,0} copy(reduce)
}
"""

hlo_module_text = b"""HloModule add_vec_module
    ENTRY %add_vec (a: s32[256], b: s32[256]) -> s32[256] {
      %a = s32[256] parameter(0)
      %b = s32[256] parameter(1)
      ROOT %sum = s32[256] add(%a, %b)
    }
    """

def hlo_compile(hlo_text):
  print("------ Going to Compile a TPU program ------\n")
  print("HLO text:")
  print(len(hlo_text))
  print(hlo_text.decode('utf8'))
  return driver_fn.TpuDriver_CompileProgramFromText(driver, hlo_text,
        1, # num_replicas
        0, # eventc
        None) # eventv
  if not cph:
    print("failed to compile!")
  else:
    return cph

cph = hlo_compile(hlo_module_reduce_text)
if not cph:
  print("failed to compile!")

cph = hlo_compile(hlo_module_text)
if not cph:
  print("failed to compile!")
else:
  compile_events = (TpuEvent_p * 1)()
  compile_events[0] = cph.contents.event

  print("------ Going to Load a TPU program ------\n")

  lph = driver_fn.TpuDriver_LoadProgram(driver,
      0, # core_id
      cph,
      1, # eventc
      compile_events) # eventv

  print(lph)
