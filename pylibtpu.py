import sys
import ctypes as _c
import dataclasses
import json
from pprint import pprint as pp

from google.protobuf.json_format import MessageToJson
import os

sys.path.append(os.path.dirname(__file__))

import tpu_driver_pb2

int8_t = _c.c_int8
int16_t = _c.c_int16
int32_t = _c.c_int32
int64_t = _c.c_int64
uint8_t = _c.c_uint8
uint16_t = _c.c_uint16
uint32_t = _c.c_uint32
uint64_t = _c.c_uint64
float32_t = _c.c_float
float64_t = _c.c_double
void_p = _c.c_void_p
char_t = _c.c_char
char_p = _c.c_char_p
bool_t = _c.c_bool

NULL = None

def arrayof(ctype, *args):
  out = (ctype * len(args))()
  for i in range(len(args)):
    out[i] = args[i]
  return out

def panic(msg):
  print(msg)
  breakpoint()

def verify(label, result):
  if not result:
    panic(f"failed to {label}!")
  return result


libtpu = _c.cdll.LoadLibrary("libtpu.so")

FuncPtr = _c.CFUNCTYPE(None)

class TpuDriver(_c.Structure):
  pass

TpuDriver_p = _c.POINTER(TpuDriver)

class TpuEvent(_c.Structure):
  pass

TpuEvent_p = _c.POINTER(TpuEvent)
TpuEventList = _c.POINTER(TpuEvent_p)

class TpuBufferHandle(_c.Structure):
  _fields_ = [
      ("internal_handle", void_p),
      ("event", TpuEvent_p),
      ("size_in_bytes", int64_t),
      ]

TpuBufferHandle_p = _c.POINTER(TpuBufferHandle)

class TpuCompiledProgramHandle(_c.Structure):
  _fields_ = [
      ("internal_handle", void_p),
      ("event", TpuEvent_p),
      ]

TpuCompiledProgramHandle_p = _c.POINTER(TpuCompiledProgramHandle)

class TpuLoadedProgramHandle(_c.Structure):
  _fields_ = [
      ("internal_handle", void_p),
      ("event", TpuEvent_p),
      ]

TpuLoadedProgramHandle_p = _c.POINTER(TpuLoadedProgramHandle)

class DeviceAssignment(_c.Structure):
  """DeviceAssignment is a serialized xla::DeviceAssignmentProto buffer."""
  _fields_ = [
      ("bytes", _c.POINTER(char_t)),
      ("size", int32_t),
      ]
  def __bytes__(self):
    if self.bytes:
      return self.bytes[0:self.size]

DeviceAssignment_p = _c.POINTER(DeviceAssignment)

class TpuAllocationShape(_c.Structure):
  _fields_ = [
      ("bytes", _c.POINTER(char_t)),
      ("size", int32_t),
      ]
  def __bytes__(self):
    if self.bytes:
      return self.bytes[0:self.size]

TpuAllocationShape_p = _c.POINTER(TpuAllocationShape)

class TpuSystemInfo(_c.Structure):
  _fields_ = [
      ("bytes", _c.POINTER(char_t)),
      ("size", int32_t),
      ]
  def __bytes__(self):
    if self.bytes:
      return self.bytes[0:self.size]

TpuSystemInfo_p = _c.POINTER(TpuSystemInfo)


@dataclasses.dataclass
class TpuStatus(_c.Structure):
  code: int
  msg: str
  _fields_ = [
      ("code", int32_t),
      ("msg", char_p),
      ]

TpuStatus_p = _c.POINTER(TpuStatus)


@dataclasses.dataclass
class TpuDriverFn(_c.Structure):
  _fields_ = [
    ("TpuDriver_Open", _c.CFUNCTYPE(TpuDriver_p, char_p)),
    ("TpuDriver_Close", _c.CFUNCTYPE(None, TpuDriver_p)),
    ("TpuDriver_Reset", _c.CFUNCTYPE(TpuStatus_p, TpuDriver_p)),
    ("TpuDriver_ComputeLinearizedBytesFromShape", FuncPtr),
    ("TpuDriver_QuerySystemInfo", _c.CFUNCTYPE(TpuSystemInfo_p, TpuDriver_p)),
    ("TpuDriver_FreeSystemInfo", _c.CFUNCTYPE(None, TpuSystemInfo_p)),
    ("TpuDriver_LinearizeShape", FuncPtr),
    ("TpuDriver_DelinearizeShape", FuncPtr),
    ("TpuDriver_CompileProgram", FuncPtr),
    ("TpuDriver_CompileProgramFromText", _c.CFUNCTYPE(TpuCompiledProgramHandle_p, TpuDriver_p, char_p, int32_t, int32_t, _c.POINTER(TpuEvent_p))),
    ("TpuDriver_FreeCompiledProgramHandle", FuncPtr),
    ("TpuDriver_LoadProgram", _c.CFUNCTYPE(TpuLoadedProgramHandle_p, TpuDriver_p, int32_t, TpuCompiledProgramHandle_p, int32_t, _c.POINTER(TpuEvent_p))),
    ("TpuDriver_UnloadProgram", FuncPtr),
    # typedef struct TpuEvent*(PrototypeTpuDriver_ExecuteProgram)(
    #     struct TpuDriver* driver, struct TpuLoadedProgramHandle* handle,
    #     int32_t inputc, struct TpuBufferHandle** input_buffer_handle,
    #     int32_t outputc, struct TpuBufferHandle** output_buffer_handle,
    #     struct DeviceAssignment device_assignment, int32_t eventc,
    #     struct TpuEvent** eventv);
    ("TpuDriver_ExecuteProgram", _c.CFUNCTYPE(TpuEvent_p,
      TpuDriver_p, TpuLoadedProgramHandle_p,
      int32_t, _c.POINTER(TpuBufferHandle_p),
      int32_t, _c.POINTER(TpuBufferHandle_p),
      DeviceAssignment, int32_t, _c.POINTER(TpuEvent_p))),
    # typedef struct TpuBufferHandle*(PrototypeTpuDriver_AllocateTuple)(
    #     struct TpuDriver* driver, int32_t core_id, int32_t memory_region,
    #     int32_t bufferc, struct TpuBufferHandle** buffer_handle, int32_t eventc,
    #     struct TpuEvent** eventv);
    ("TpuDriver_AllocateTuple", _c.CFUNCTYPE(TpuBufferHandle_p,
      TpuDriver_p, int32_t, int32_t, int32_t, _c.POINTER(TpuBufferHandle_p), int32_t, _c.POINTER(TpuEvent_p))),
    # typedef struct TpuBufferHandle*(PrototypeTpuDriver_Allocate)(
    #     struct TpuDriver* driver, int32_t core_id, int32_t memory_region,
    #     int64_t num_bytes, int32_t eventc, struct TpuEvent** eventv);
    ("TpuDriver_Allocate", _c.CFUNCTYPE(TpuBufferHandle_p,
      TpuDriver_p, int32_t, int32_t, int64_t, int32_t, _c.POINTER(TpuEvent_p))),
    # typedef struct TpuBufferHandle*(PrototypeTpuDriver_AllocateShape)(
    #     struct TpuDriver* driver, int32_t core_id, int32_t memory_region,
    #     const struct TpuAllocationShape shape, int32_t eventc,
    #     struct TpuEvent** eventv);
    ("TpuDriver_AllocateShape", _c.CFUNCTYPE(TpuBufferHandle_p,
      TpuDriver_p, int32_t, int32_t, TpuAllocationShape, int32_t, _c.POINTER(TpuEvent_p))),
    # /* Note: We are not responsible for freeing the event within the
    #  * TpuBufferHandle. You have to call FreeEvent separately to ensure that memory
    #  * does not leak.
    #  */
    # typedef struct TpuEvent*(PrototypeTpuDriver_Deallocate)(
    #     struct TpuDriver* driver, struct TpuBufferHandle* buffer_handle,
    #     int32_t eventc, struct TpuEvent** eventv);
    ("TpuDriver_Deallocate", _c.CFUNCTYPE(TpuEvent_p,
      TpuDriver_p, TpuBufferHandle_p, int32_t, _c.POINTER(TpuEvent_p))),
    ("TpuDriver_TransferToDevice", FuncPtr),
    ("TpuDriver_TransferFromDevice", FuncPtr),
    ("TpuDriver_TransferFromDeviceToDevice", FuncPtr),
    ("TpuDriver_GetCompiledProgramShape", FuncPtr),
    ("TpuDriver_FreeCompiledProgramShape", FuncPtr),
    ("TpuDriver_EventAddCallback", FuncPtr),
    ("TpuDriver_EventAwait", FuncPtr),
    ("TpuDriver_FreeEvent", FuncPtr),
    ("TpuDriver_FreeStatus", FuncPtr),
    ("TpuDriver_Version", _c.CFUNCTYPE(char_p)),
  ]
  @property
  def version(self) -> str:
    if self.TpuDriver_Version:
      return self.TpuDriver_Version().decode('utf8')
  def __repr__(self):
    return f"{type(self).__name__}(version={self.version!r})"



libtpu.TpuDriver_Initialize.argtypes = [_c.POINTER(TpuDriverFn), bool_t]
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


def hlo_compile(hlo_text):
  print("------ Going to Compile a TPU program ------\n")
  print("HLO text:")
  print(len(hlo_text))
  print(hlo_text.decode('utf8'))
  return driver_fn.TpuDriver_CompileProgramFromText(driver, hlo_text,
        1, # num_replicas
        0, # eventc
        None) # eventv

def hlo_assert_compile(hlo_text):
  return verify("compile", hlo_compile(hlo_text))


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

cph = hlo_assert_compile(hlo_module_reduce_text)


# typedef float dtype_t;
# const int numel = 16*1024*1024;
# const int size = sizeof(dtype_t) * numel;
dtype_t = float32_t
numel = 16 * 1024 * 1024
size = _c.sizeof(dtype_t) * numel
dtype_s = f"f32[{numel}]"
hlo_module_text = f"""HloModule add_vec_module
    ENTRY %add_vec (a: {dtype_s}, b: {dtype_s}) -> {dtype_s} {{
      %a = {dtype_s} parameter(0)
      %b = {dtype_s} parameter(1)
      ROOT %sum = {dtype_s} add(%a, %b)
    }}
    """.encode('utf8')

cph = hlo_assert_compile(hlo_module_text)

print("------ Going to Load a TPU program ------\n")
compile_events = arrayof(TpuEvent_p, cph.contents.event)
lph = driver_fn.TpuDriver_LoadProgram(driver,
    0, # core_id
    cph,
    1, # eventc
    compile_events) # eventv
verify("load", lph)


print("------ Going to Allocate a TPU Buffer ------\n")
verify("allocate", buf_a_handle := driver_fn.TpuDriver_Allocate(driver,
    0, # core_id
    1, # memory_region
    size, # bytes
    0, # eventc
    NULL)) # eventv

print("------ Going to Allocate a TPU Buffer ------\n")
verify("allocate", buf_b_handle := driver_fn.TpuDriver_Allocate(driver,
    0, # core_id
    1, # memory_region
    size, # bytes
    0, # eventc
    NULL)) # eventv

print("------ Going to Allocate a TPU Buffer ------\n")
verify("allocate", buf_sum_handle := driver_fn.TpuDriver_Allocate(driver,
    0, # core_id
    1, # memory_region
    size, # bytes
    0, # eventc
    NULL)) # eventv

a_src = (dtype_t * numel)()
b_src = (dtype_t * numel)()
sum_src = (dtype_t * numel)()
a_src[:] = [1] * numel
b_src[:] = [2] * numel
sum_src[:] = [0] * numel

print("------ Going to Transfer To Device ------\n")
# TpuEvent* allocate_buf_a_events[] = {buf_a_handle->event};
allocate_buf_a_events = arrayof(TpuEvent_p, buf_a_handle.contents.event)
# struct TpuEvent* transfer_ev1 =
#     driver_fn.TpuDriver_TransferToDevice(driver, a_src, buf_a_handle,
#       /*eventc=*/1, /*eventv=*/allocate_buf_a_events);
verify("transfer", transfer_ev1 := driver_fn.TpuDriver_TransferToDevice(driver, a_src, buf_a_handle,
    1, # eventc
    allocate_buf_a_events)) # eventv
    

print("------ Going to Transfer To Device ------\n")
# TpuEvent* allocate_buf_b_events[] = {buf_b_handle->event};
allocate_buf_b_events = arrayof(TpuEvent_p, buf_b_handle.contents.event)
# struct TpuEvent* transfer_ev2 =
#     driver_fn.TpuDriver_TransferToDevice(driver, b_src, buf_b_handle,
#       /*eventc=*/1, /*eventv=*/allocate_buf_b_events);
verify("transfer", transfer_ev2 := driver_fn.TpuDriver_TransferToDevice(driver, b_src, buf_b_handle,
    1, # eventc
    allocate_buf_b_events)) # eventv
    

# fprintf(stdout, "------ Going to Execute a TPU program ------\n"); fflush(stdout);
print("------ Going to Execute a TPU program ------\n")
# DeviceAssignment device_assignment = {NULL, 0};
device_assignment = DeviceAssignment()
# TpuBufferHandle* input_buffer_handle[] = {buf_a_handle, buf_b_handle};
input_buffer_handle = arrayof(TpuBufferHandle_p, buf_a_handle, buf_b_handle)
# TpuBufferHandle* output_buffer_handle[] = {buf_sum_handle};
output_buffer_handle = arrayof(TpuBufferHandle_p, buf_sum_handle)
# TpuEvent* transfer_events[] = {transfer_ev1, transfer_ev2};
transfer_events = arrayof(TpuEvent_p, transfer_ev1, transfer_ev2)
# struct TpuEvent* execute_event =
#     driver_fn.TpuDriver_ExecuteProgram(driver, lph,
#     /*inputc=*/2, /*input_buffer_handle=*/input_buffer_handle,
#     /*outputc=*/1, /*output_buffer_handle=*/output_buffer_handle,
#     device_assignment,
#     /*eventc=*/2, /*eventv*/transfer_events);
verify("execute", execute_event := driver_fn.TpuDriver_ExecuteProgram(driver, lph,
    2, input_buffer_handle,
    1, output_buffer_handle,
    device_assignment,
    2, transfer_events))

