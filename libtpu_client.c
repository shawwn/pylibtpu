/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Before you start, make sure libtpu.so, libtpu.h and libtpu_client.c are in
// the same working directory.
//
// To compile: gcc -o libtpu_client libtpu_client.c -ldl
// To run: sudo ./libtpu_client

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libtpu.h"

void* LoadAndInitializeDriver(const char* shared_lib,
                              struct TpuDriverFn* driver_fn) {
  void* handle;
  handle = dlopen(shared_lib, RTLD_NOW);
  if (!handle) {
    fprintf(stderr, "Error: %s\n", dlerror());
    exit(EXIT_FAILURE);
  }

  PrototypeTpuDriver_Initialize* initialize_fn;
  *(void**)(&initialize_fn) = dlsym(handle, "TpuDriver_Initialize");
  initialize_fn(driver_fn, true);

  return handle;
}

int main(int argc, char** argv) {
  char* api_path = "libtpu.so";
  if (argc == 2) {
    api_path = argv[1];
  }

  struct TpuDriverFn driver_fn;
  void* handle = LoadAndInitializeDriver(api_path, &driver_fn);

  fprintf(stdout, "------ Going to Query Version ------\n");
  fprintf(stdout, "TPU Driver Version: %s\n", driver_fn.TpuDriver_Version());

  fprintf(stdout, "------ Going to Open a TPU Driver ------\n");
  struct TpuDriver* driver = driver_fn.TpuDriver_Open("local://");

  fprintf(stdout, "------ Going to Query for System Information ------\n");
  struct TpuSystemInfo* info = driver_fn.TpuDriver_QuerySystemInfo(driver);
  driver_fn.TpuDriver_FreeSystemInfo(info);

  // An example of simple program to sum two parameters.
  const char* hlo_module_text = R"(HloModule add_vec_module
    ENTRY %add_vec (a: f32[16777216], b: f32[16777216]) -> f32[16777216] {
      %a = f32[16777216] parameter(0)
      %b = f32[16777216] parameter(1)
      ROOT %sum = f32[16777216] add(%a, %b)
    }
    )";

  fprintf(stdout, "------ Going to Compile a TPU program ------\n"); fflush(stdout);
  printf("HLO text:\n%d\n%s\n", (int)strlen(hlo_module_text), hlo_module_text);
  struct TpuCompiledProgramHandle* cph =
      driver_fn.TpuDriver_CompileProgramFromText(driver, hlo_module_text,
      /*num_replicas=*/1, /*eventc=*/0, /*eventv*/NULL);
  printf("cph = 0x%p\n", cph);

  TpuEvent* compile_events[] = {cph->event};
  fprintf(stdout, "------ Going to Load a TPU program ------\n"); fflush(stdout);
  struct TpuLoadedProgramHandle* lph =
      driver_fn.TpuDriver_LoadProgram(driver, /*core_id=*/0, cph,
      /*eventc=*/1, /*eventv=*/compile_events);

  typedef float dtype_t;
  const int numel = 16*1024*1024;
  const int size = sizeof(dtype_t) * numel;

  fprintf(stdout, "------ Going to Allocate a TPU Buffer ------\n"); fflush(stdout);
  struct TpuBufferHandle* buf_a_handle =
      driver_fn.TpuDriver_Allocate(driver, /*core-id=*/0, /*memory_region=*/1,
        /*bytes=*/size, /*eventc=*/0, /*eventv=*/NULL);
  fprintf(stdout, "------ Going to Allocate a TPU Buffer ------\n"); fflush(stdout);
  struct TpuBufferHandle* buf_b_handle =
      driver_fn.TpuDriver_Allocate(driver, /*core-id=*/0, /*memory_region=*/1,
        /*bytes=*/size, /*eventc=*/0, /*eventv=*/NULL);
  fprintf(stdout, "------ Going to Allocate a TPU Buffer ------\n"); fflush(stdout);
  struct TpuBufferHandle* buf_sum_handle =
      driver_fn.TpuDriver_Allocate(driver, /*core-id=*/0, /*memory_region=*/1,
        /*bytes=*/size, /*eventc=*/0, /*eventv=*/NULL);

  dtype_t* a_src = calloc(sizeof(dtype_t), numel);
  dtype_t* b_src = calloc(sizeof(dtype_t), numel);
  dtype_t* sum_src = calloc(sizeof(dtype_t), numel);
  for (int i = 0; i < numel; ++i) {
    a_src[i] = 1;
    b_src[i] = 2;
    sum_src[i] = 0;
  }

  TpuEvent* allocate_buf_a_events[] = {buf_a_handle->event};
  fprintf(stdout, "------ Going to Transfer To Device ------\n"); fflush(stdout);
  struct TpuEvent* transfer_ev1 =
      driver_fn.TpuDriver_TransferToDevice(driver, a_src, buf_a_handle,
        /*eventc=*/1, /*eventv=*/allocate_buf_a_events);
  TpuEvent* allocate_buf_b_events[] = {buf_a_handle->event};
  fprintf(stdout, "------ Going to Transfer To Device ------\n"); fflush(stdout);
  struct TpuEvent* transfer_ev2 =
      driver_fn.TpuDriver_TransferToDevice(driver, b_src, buf_b_handle,
        /*eventc=*/1, /*eventv=*/allocate_buf_b_events);

  fprintf(stdout, "------ Going to Execute a TPU program ------\n"); fflush(stdout);
  DeviceAssignment device_assignment = {NULL, 0};
  TpuBufferHandle* input_buffer_handle[] = {buf_a_handle, buf_b_handle};
  TpuBufferHandle* output_buffer_handle[] = {buf_sum_handle};
  TpuEvent* transfer_events[] = {transfer_ev1, transfer_ev2};
  struct TpuEvent* execute_event =
      driver_fn.TpuDriver_ExecuteProgram(driver, lph,
      /*inputc=*/2, /*input_buffer_handle=*/input_buffer_handle,
      /*outputc=*/1, /*output_buffer_handle=*/output_buffer_handle,
      device_assignment,
      /*eventc=*/2, /*eventv*/transfer_events);

  fprintf(stdout, "------ Going to Transfer From Device ------\n"); fflush(stdout);
  TpuEvent* execute_events[] = {execute_event};
  struct TpuEvent* transfer_sum_event =
      driver_fn.TpuDriver_TransferFromDevice(driver, buf_sum_handle, sum_src,
        /*eventc=*/1, /*eventv=*/execute_events);

  TpuStatus* status = driver_fn.TpuDriver_EventAwait(transfer_sum_event,
                                                     10000000);
  if (status->code != 0) {
    fprintf(stdout, "Transfer Event Await: Code: %d, Message: %s\n",
          status->code, status->msg);
    fflush(stdout);
  }

  fprintf(stdout, "------ Going to Unload a TPU program ------\n"); fflush(stdout);
  struct TpuEvent* unload_program_event = driver_fn.TpuDriver_UnloadProgram(
      driver, lph, /*eventc=*/1, /*eventv=*/execute_events);

  fprintf(stdout, "------ Going to Deallocate a TPU Buffer ------\n"); fflush(stdout);
  struct TpuEvent* dealloc_ev1 = driver_fn.TpuDriver_Deallocate(driver,
      buf_a_handle, /*eventc=*/0, /*eventv=*/NULL);
  driver_fn.TpuDriver_FreeEvent(dealloc_ev1);

  fprintf(stdout, "------ Going to Deallocate a TPU Buffer ------\n"); fflush(stdout);
  struct TpuEvent* dealloc_ev2 = driver_fn.TpuDriver_Deallocate(driver,
      buf_b_handle, /*eventc=*/0, /*eventv=*/NULL);
  driver_fn.TpuDriver_FreeEvent(dealloc_ev2);

  fprintf(stdout, "------ Going to Deallocate a TPU Buffer ------\n"); fflush(stdout);
  struct TpuEvent* dealloc_ev3 = driver_fn.TpuDriver_Deallocate(driver,
      buf_sum_handle, /*eventc=*/0, /*eventv=*/NULL);
  driver_fn.TpuDriver_FreeEvent(dealloc_ev3);

  fprintf(stdout, "sum:\n"); fflush(stdout);
  double total = 0;
  size_t i;
  for (i = 0; i < numel; ++i) {
    if (numel < 1048576) {
      fprintf(stdout, "%f ", sum_src[i]);
    }
    total += sum_src[i];
  }
  fprintf(stdout, "...\n"); fflush(stdout);
  fprintf(stdout, "shape: f32[16777216] (numel=%d)\n", numel); fflush(stdout);
  //fprintf(stdout, "total: %jd\n", total); fflush(stdout);
  fprintf(stdout, "total: %f\n", total); fflush(stdout);

  dlclose(handle);
  exit(EXIT_SUCCESS);
}
