// ane_bridge.h — C-callable bridge to ANE private APIs for the bounded
// feasibility probe. Adapted from maderix/ANE bridge/ane_bridge.h and
// bridge/ane_bridge.m (MIT). See LICENSE-ANE.md for attribution and the
// pinned upstream commit d91c9845c0784dec7753048954fc6d0e8411fe29.
//
// This is a STRICT, bounded subset of the upstream bridge:
//   * dtype fp16 only, single input / single output tensor
//   * explicit IOSurface sizing (width=row bytes, height=1, bpe=1, pixel
//     format 0), matching upstream inmem_basic.m / ane_bridge.m layout
//   * every private-API call is wrapped in @try/@catch, nil-checked, and
//     checked for NSError; failures are surfaced to the caller, never
//     silently ignored
//   * compile and load are exposed as SEPARATE calls so the controller can
//     record statuses and timings independently (compile vs dispatch)
//   * a process-wide compile-attempt budget (<= MAX_COMPILE_ATTEMPTS),
//     counted across BOTH successes and failures
//   * the per-model temp dir (NSTemporaryDirectory()/<hexId>, exactly as
//     upstream) is PRESERVED after compile so the compiler-produced
//     artifacts can be inventoried/hashed; the controller owns cleanup
//
// Phase-1 boundary: this header + implementation are written and host-
// compiled, but compile/load/evaluate are only reached from the probe's
// --mode compile-and-run path, which the controller does not invoke until
// phase 2 is authorized.

#ifndef ANE_BRIDGE_H
#define ANE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ANE_MAX_COMPILE_ATTEMPTS 8

// Opaque compiled-kernel handle.
typedef struct ANEKernelHandle ANEKernelHandle;

typedef struct {
    bool framework_loaded;
    bool cls_client;
    bool cls_descriptor;
    bool cls_inmem;
    bool cls_request;
    bool cls_io;
    // Class methods (queried with +respondsToSelector:).
    bool m_modelWithMILText_weights_optionsPlist; // _ANEInMemoryModelDescriptor
    bool m_inMemoryModelWithDescriptor;           // _ANEInMemoryModel
    // Instance methods (queried with +instanceRespondsToSelector:).
    bool m_hexStringIdentifier;
    bool m_compileWithQoS_options_error;
    bool m_loadWithQoS_options_error;
    bool m_evaluateWithQoS_options_request_error;
    bool m_unloadWithQoS_error;
    bool m_state;
    char framework_path[1024];
} ANEBridgeAvailability;

// Resolve private framework/classes/methods once (idempotent). Always fills
// *availability when non-NULL, even on repeat calls. Returns 0 on success
// (classes resolved), -1 on failure.
int ane_bridge_init(ANEBridgeAvailability *availability);

// Compile a MIL program with one weight blob into an in-memory ANE model.
// Does NOT load (load is a separate, timed call). On success returns a
// handle; on failure returns NULL and, if *err is non-NULL, writes a UTF-8
// message (capped at err_cap bytes).
//
// mil_text     : UTF-8 MIL program text (NUL-terminated)
// weight_data  : raw weight blob (128-byte header + fp16 data), may be NULL
// weight_len   : length of weight_data
// in_bytes     : logical input tensor byte size (fp16)
// out_bytes    : logical output tensor byte size (fp16)
// stride_bytes : padded IOSurface row/alloc byte size (must be >= in/out)
ANEKernelHandle *ane_bridge_compile(const char *mil_text,
                                     const uint8_t *weight_data, size_t weight_len,
                                     size_t in_bytes, size_t out_bytes,
                                     size_t stride_bytes,
                                     char *err, size_t err_cap);

// Load a previously compiled model. Separate from compile. 0 on success.
int ane_bridge_load(ANEKernelHandle *kernel, char *err, size_t err_cap);

// Evaluate (dispatch) the loaded model once. 0 on success.
int ane_bridge_eval(ANEKernelHandle *kernel, char *err, size_t err_cap);

// Write input tensor bytes (fp16) into the input IOSurface. Returns 0 on
// success, -1 if bytes != expected input size (does NOT truncate).
int ane_bridge_write_input(ANEKernelHandle *kernel, const void *data, size_t bytes);

// Read output tensor bytes (fp16) from the output IOSurface. Returns 0 on
// success, -1 if bytes != expected output size (does NOT truncate).
int ane_bridge_read_output(ANEKernelHandle *kernel, void *data, size_t bytes);

// Query model state after load. Returns -1 if unavailable.
long ane_bridge_state(ANEKernelHandle *kernel);

// Return the per-model temp dir path (NSTemporaryDirectory()/<hexId>) that
// the compiler uses. Owned by the bridge; valid until ane_bridge_free.
const char *ane_bridge_tmpdir(ANEKernelHandle *kernel);

// Return the model's hex-string identifier. Owned by the bridge.
const char *ane_bridge_identifier(ANEKernelHandle *kernel);

// Number of IOSurface alloc bytes actually allocated for the given stride.
size_t ane_bridge_surface_alloc_bytes(ANEKernelHandle *kernel);

// Unload the model and free all resources. Does NOT delete the temp dir
// (compiler artifacts are preserved; the controller owns cleanup).
void ane_bridge_free(ANEKernelHandle *kernel);

// Process-wide compile-attempt counter (successes AND failures).
int ane_bridge_get_compile_attempts(void);

// Build an FP16 weight blob in the ANE layout used by the reference:
//   64-byte "global" header (byte0=0x01, byte4=0x02)
//   64-byte "chunk"  header (bytes64-67=0xEF 0xBE 0xAD 0xDE, byte68=0x01,
//                            uint32@72=data_size, uint32@80=128)
//   fp16 data at byte 128
// src is row-major [out_ch * in_ch] float; returns allocated buffer with
// *out_len set. Caller must free() the returned buffer.
uint8_t *ane_bridge_build_weight_blob(const float *src, size_t n_elems, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif // ANE_BRIDGE_H
