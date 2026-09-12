// ane_bridge.m — Objective-C implementation of the ANE private-API bridge.
//
// ADAPTED from maderix/ANE bridge/ane_bridge.m and inmem_basic.m (MIT,
// pinned commit d91c9845c0784dec7753048954fc6d0e8411fe29). See
// LICENSE-ANE.md for full attribution. The MIL text, weight-blob byte layout,
// IOSurface sizing, per-model temp-dir convention, and objc_msgSend call
// conventions are copied from the upstream source. Added (not in upstream):
// separate compile/load, @try/@catch + NSError capture, strict validation,
// instance-method availability probing (via instancesRespondToSelector:),
// explicit retained model/request ownership, compile-attempt budget, and
// artifact preservation.

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include "ane_bridge.h"

// --- Private class references ---
static Class g_ANEDesc  = nil;
static Class g_ANEInMem = nil;
static Class g_ANEReq   = nil;
static Class g_ANEIO    = nil;
static bool g_initialized = false;
static int  g_compile_attempts = 0;

#define ANE_FRAMEWORK_PATH \
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"

struct ANEKernelHandle {
    CFTypeRef model;           // CFBridgingRetain(_ANEInMemoryModel)
    CFTypeRef request;         // CFBridgingRetain(_ANERequest)
    IOSurfaceRef ioInput;      // input IOSurface
    IOSurfaceRef ioOutput;     // output IOSurface
    char *tmpDir;              // NSTemporaryDirectory()/<hexId>
    char *identifier;          // hexStringIdentifier
    size_t inBytes;            // logical input bytes (fp16)
    size_t outBytes;           // logical output bytes (fp16)
    size_t strideBytes;        // padded IOSurface alloc/row bytes
};

static inline id model_obj(ANEKernelHandle *k) { return (__bridge id)k->model; }
static inline id request_obj(ANEKernelHandle *k) { return (__bridge id)k->request; }
// Build an error string from an optional message + NSError + NSException.
// Only uses NSString formatting (%@) and then copies via %s, so snprintf
// never sees an ObjC format specifier. Guards err/err_cap.
static void set_err(char *err, size_t err_cap,
                    const char *msg, NSError *e, NSException *ex) {
    if (!err || err_cap == 0) return;
    NSMutableString *s = [NSMutableString stringWithUTF8String:
                            (msg && msg[0]) ? msg : "ane_bridge error"];
    if (e)  [s appendFormat:@" NSError=%@", [e description]];
    if (ex) [s appendFormat:@" NSException=%@", [ex description]];
    const char *c = [s UTF8String];
    snprintf(err, err_cap, "%s", c ? c : "");
}

int ane_bridge_init(ANEBridgeAvailability *availability) {
    // Resolve classes FIRST so availability reflects reality on first call.
    if (!g_initialized) {
        dlopen(ANE_FRAMEWORK_PATH, RTLD_NOW);
        g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
        g_ANEReq   = NSClassFromString(@"_ANERequest");
        g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
        g_initialized = (g_ANEDesc && g_ANEInMem && g_ANEReq && g_ANEIO);
    }

    // Always (re)populate availability on every call, idempotent or not.
    if (availability) {
        memset(availability, 0, sizeof(*availability));
        availability->framework_loaded =
            (dlopen(ANE_FRAMEWORK_PATH, RTLD_NOW) != NULL);
        availability->cls_client     = (NSClassFromString(@"_ANEClient") != nil);
        availability->cls_descriptor = (g_ANEDesc != nil);
        availability->cls_inmem      = (g_ANEInMem != nil);
        availability->cls_request    = (g_ANEReq != nil);
        availability->cls_io         = (g_ANEIO != nil);
        snprintf(availability->framework_path, sizeof(availability->framework_path),
                 "%s", ANE_FRAMEWORK_PATH);
        // Class methods (queried with +respondsToSelector:).
        availability->m_modelWithMILText_weights_optionsPlist =
            (g_ANEDesc && [g_ANEDesc respondsToSelector:@selector(modelWithMILText:weights:optionsPlist:)]);
        availability->m_inMemoryModelWithDescriptor =
            (g_ANEInMem && [g_ANEInMem respondsToSelector:@selector(inMemoryModelWithDescriptor:)]);
        // Instance methods (queried with +instancesRespondToSelector:).
        availability->m_hexStringIdentifier =
            (g_ANEInMem && [g_ANEInMem instancesRespondToSelector:@selector(hexStringIdentifier)]);
        availability->m_compileWithQoS_options_error =
            (g_ANEInMem && [g_ANEInMem instancesRespondToSelector:@selector(compileWithQoS:options:error:)]);
        availability->m_loadWithQoS_options_error =
            (g_ANEInMem && [g_ANEInMem instancesRespondToSelector:@selector(loadWithQoS:options:error:)]);
        availability->m_evaluateWithQoS_options_request_error =
            (g_ANEInMem && [g_ANEInMem instancesRespondToSelector:@selector(evaluateWithQoS:options:request:error:)]);
        availability->m_unloadWithQoS_error =
            (g_ANEInMem && [g_ANEInMem instancesRespondToSelector:@selector(unloadWithQoS:error:)]);
        availability->m_state =
            (g_ANEInMem && [g_ANEInMem instancesRespondToSelector:@selector(state)]);
    }

    if (!g_initialized) {
        fprintf(stderr, "ane_bridge: failed to resolve ANE private classes\n");
        return -1;
    }
    return 0;
}

static IOSurfaceRef create_surface(size_t bytes, char *err, size_t err_cap) {
    // Matches upstream inmem_basic.m / ane_bridge.m IOSurface creation:
    // width = byte count, height 1, bpe 1, row = byte count, pixel format 0.
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
    if (!s) {
        set_err(err, err_cap, "IOSurfaceCreate returned NULL", nil, nil);
        return NULL;
    }
    if (IOSurfaceGetAllocSize(s) < bytes) {
        set_err(err, err_cap, "IOSurface alloc size < requested bytes", nil, nil);
        CFRelease(s);
        return NULL;
    }
    if (IOSurfaceGetBaseAddress(s) == NULL) {
        set_err(err, err_cap, "IOSurface base address is NULL", nil, nil);
        CFRelease(s);
        return NULL;
    }
    return s;
}

ANEKernelHandle *ane_bridge_compile(const char *mil_text,
                                     const uint8_t *weight_data, size_t weight_len,
                                     size_t in_bytes, size_t out_bytes,
                                     size_t stride_bytes,
                                     char *err, size_t err_cap)
{
    @autoreleasepool {
        if (!g_initialized) {
            set_err(err, err_cap, "ane_bridge: not initialized", nil, nil);
            return NULL;
        }
        // Strict validation BEFORE any private-API call.
        if (!mil_text || strlen(mil_text) == 0) {
            set_err(err, err_cap, "empty MIL text", nil, nil);
            return NULL;
        }
        if (in_bytes == 0 || out_bytes == 0 || stride_bytes == 0 ||
            stride_bytes < in_bytes || stride_bytes < out_bytes) {
            set_err(err, err_cap, "invalid in/out/stride byte bounds", nil, nil);
            return NULL;
        }
        if (weight_data && weight_len == 0) {
            set_err(err, err_cap, "weight data with zero length", nil, nil);
            return NULL;
        }
        // Compile-attempt budget, counting successes AND failures.
        if (g_compile_attempts >= ANE_MAX_COMPILE_ATTEMPTS) {
            set_err(err, err_cap, "compile-attempt budget exhausted", nil, nil);
            return NULL;
        }
        g_compile_attempts++;

        NSData *milData = [NSData dataWithBytes:mil_text length:strlen(mil_text)];
        NSError *e = nil;

        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        if (weight_data && weight_len > 0) {
            wdict[@"@model_path/weights/weight.bin"] =
                @{@"offset": @0, @"data": [NSData dataWithBytes:weight_data
                                                         length:weight_len]};
        }

        id desc = nil, mdl = nil;
        @try {
            desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                milData, wdict, @{});
        } @catch (NSException *ex) {
            set_err(err, err_cap, "modelWithMILText raised", nil, ex);
            return NULL;
        }
        if (!desc) {
            set_err(err, err_cap, "modelWithMILText returned nil", nil, nil);
            return NULL;
        }

        @try {
            mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
                g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        } @catch (NSException *ex) {
            set_err(err, err_cap, "inMemoryModelWithDescriptor raised", nil, ex);
            return NULL;
        }
        if (!mdl) {
            set_err(err, err_cap, "inMemoryModelWithDescriptor returned nil", nil, nil);
            return NULL;
        }

        // Exact upstream temp-dir convention: NSTemporaryDirectory()/<hexId>.
        // The controller launches this process with TMPDIR pointing at a
        // task-owned run/framework_tmp dir, so Foundation's temp dir is
        // already isolated. We validate the identifier is a plain hex string
        // and that the resulting path stays scoped under the temp dir.
        NSString *hexId = nil;
        @try {
            hexId = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        } @catch (NSException *ex) {
            set_err(err, err_cap, "hexStringIdentifier raised", nil, ex);
            return NULL;
        }
        if (![hexId isKindOfClass:[NSString class]] || hexId.length == 0) {
            set_err(err, err_cap, "invalid hex identifier", nil, nil);
            return NULL;
        }
        NSCharacterSet *hex = [NSCharacterSet characterSetWithCharactersInString:
            @"0123456789abcdefABCDEF"];
        if ([hexId rangeOfCharacterFromSet:[hex invertedSet]].location != NSNotFound) {
            set_err(err, err_cap, "hex identifier contains non-hex chars", nil, nil);
            return NULL;
        }
        NSString *tmpRoot = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSString *resolved = [tmpRoot stringByStandardizingPath];
        NSString *tmpRootStd = [NSTemporaryDirectory() stringByStandardizingPath];
        if (![resolved hasPrefix:tmpRootStd]) {
            set_err(err, err_cap, "temp dir escapes NSTemporaryDirectory", nil, nil);
            return NULL;
        }

        NSFileManager *fm = [NSFileManager defaultManager];
        NSError *fsErr = nil;
        NSString *weightsDir = [tmpRoot stringByAppendingPathComponent:@"weights"];
        if (![fm createDirectoryAtPath:weightsDir withIntermediateDirectories:YES
                            attributes:nil error:&fsErr]) {
            set_err(err, err_cap, "could not create weights dir", fsErr, nil);
            return NULL;
        }
        if (![milData writeToFile:[tmpRoot stringByAppendingPathComponent:@"model.mil"]
                       atomically:YES]) {
            set_err(err, err_cap, "could not write model.mil", nil, nil);
            return NULL;
        }
        if (weight_data && weight_len > 0) {
            if (![[NSData dataWithBytes:weight_data length:weight_len]
                     writeToFile:[tmpRoot stringByAppendingPathComponent:@"weights/weight.bin"]
                       atomically:YES]) {
                set_err(err, err_cap, "could not write weights/weight.bin", nil, nil);
                return NULL;
            }
        }

        BOOL compiled = NO;
        @try {
            compiled = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        } @catch (NSException *ex) {
            set_err(err, err_cap, "compileWithQoS raised", nil, ex);
            return NULL; // artifacts preserved
        }
        if (!compiled) {
            set_err(err, err_cap, "ANE compile failed", e, nil);
            return NULL; // artifacts preserved
        }

        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        if (!k) {
            set_err(err, err_cap, "calloc failed", nil, nil);
            return NULL;
        }
        k->inBytes = in_bytes;
        k->outBytes = out_bytes;
        k->strideBytes = stride_bytes;
        k->tmpDir = strdup([resolved UTF8String]);
        k->identifier = strdup([hexId UTF8String]);
        k->model = CFBridgingRetain(mdl);          // explicit retained ownership
        k->ioInput = create_surface(stride_bytes, err, err_cap);
        if (!k->ioInput) {
            CFBridgingRelease(k->model);
            free(k->tmpDir); free(k->identifier); free(k);
            return NULL;
        }
        k->ioOutput = create_surface(stride_bytes, err, err_cap);
        if (!k->ioOutput) {
            CFRelease(k->ioInput); CFBridgingRelease(k->model);
            free(k->tmpDir); free(k->identifier); free(k);
            return NULL;
        }

        id wIn = nil, wOut = nil;
        @try {
            wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInput);
            wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutput);
        } @catch (NSException *ex) {
            set_err(err, err_cap, "objectWithIOSurface raised", nil, ex);
            CFRelease(k->ioInput); CFRelease(k->ioOutput); CFBridgingRelease(k->model);
            free(k->tmpDir); free(k->identifier); free(k);
            return NULL;
        }
        if (!wIn || !wOut) {
            set_err(err, err_cap, "objectWithIOSurface returned nil", nil, nil);
            CFRelease(k->ioInput); CFRelease(k->ioOutput); CFBridgingRelease(k->model);
            free(k->tmpDir); free(k->identifier); free(k);
            return NULL;
        }

        @try {
            k->request = CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                g_ANEReq,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0));
        } @catch (NSException *ex) {
            set_err(err, err_cap, "requestWithInputs raised", nil, ex);
            CFRelease(k->ioInput); CFRelease(k->ioOutput); CFBridgingRelease(k->model);
            free(k->tmpDir); free(k->identifier); free(k);
            return NULL;
        }
        if (!k->request) {
            set_err(err, err_cap, "request construction returned nil", nil, nil);
            CFRelease(k->ioInput); CFRelease(k->ioOutput); CFBridgingRelease(k->model);
            free(k->tmpDir); free(k->identifier); free(k);
            return NULL;
        }

        return k;
    }
}

int ane_bridge_load(ANEKernelHandle *kernel, char *err, size_t err_cap) {
    @autoreleasepool {
        if (!kernel || !kernel->model) {
            set_err(err, err_cap, "load on null handle", nil, nil);
            return -1;
        }
        NSError *e = nil;
        BOOL ok = NO;
        @try {
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                model_obj(kernel), @selector(loadWithQoS:options:error:), 21, @{}, &e);
        } @catch (NSException *ex) {
            set_err(err, err_cap, "loadWithQoS raised", nil, ex);
            return -1;
        }
        if (!ok) {
            set_err(err, err_cap, "ANE load failed", e, nil);
            return -1;
        }
        return 0;
    }
}

int ane_bridge_eval(ANEKernelHandle *kernel, char *err, size_t err_cap) {
    @autoreleasepool {
        if (!kernel || !kernel->model) {
            set_err(err, err_cap, "eval on null handle", nil, nil);
            return -1;
        }
        NSError *e = nil;
        BOOL ok = NO;
        @try {
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model_obj(kernel), @selector(evaluateWithQoS:options:request:error:),
                21, @{}, request_obj(kernel), &e);
        } @catch (NSException *ex) {
            set_err(err, err_cap, "evaluateWithQoS raised", nil, ex);
            return -1;
        }
        if (!ok) {
            set_err(err, err_cap, "ANE evaluate failed", e, nil);
            return -1;
        }
        return 0;
    }
}

int ane_bridge_write_input(ANEKernelHandle *kernel, const void *data, size_t bytes) {
    if (!kernel || !data) return -1;
    if (bytes != kernel->inBytes) return -1; // refuse silent truncation
    if (IOSurfaceGetBaseAddress(kernel->ioInput) == NULL) return -1;
    if (IOSurfaceLock(kernel->ioInput, 0, NULL) != kIOReturnSuccess) return -1;
    memcpy(IOSurfaceGetBaseAddress(kernel->ioInput), data, bytes);
    return (IOSurfaceUnlock(kernel->ioInput, 0, NULL) == kIOReturnSuccess) ? 0 : -1;
}

int ane_bridge_read_output(ANEKernelHandle *kernel, void *data, size_t bytes) {
    if (!kernel || !data) return -1;
    if (bytes != kernel->outBytes) return -1; // refuse silent truncation
    if (IOSurfaceGetBaseAddress(kernel->ioOutput) == NULL) return -1;
    if (IOSurfaceLock(kernel->ioOutput, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return -1;
    memcpy(data, IOSurfaceGetBaseAddress(kernel->ioOutput), bytes);
    return (IOSurfaceUnlock(kernel->ioOutput, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess) ? 0 : -1;
}

long ane_bridge_state(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel || !kernel->model) return -1;
        if (![model_obj(kernel) respondsToSelector:@selector(state)]) return -1;
        @try {
            return (long)((NSUInteger(*)(id,SEL))objc_msgSend)(
                model_obj(kernel), @selector(state));
        } @catch (NSException *ex) {
            return -1;
        }
    }
}

const char *ane_bridge_tmpdir(ANEKernelHandle *kernel) {
    return kernel ? kernel->tmpDir : NULL;
}

const char *ane_bridge_identifier(ANEKernelHandle *kernel) {
    return kernel ? kernel->identifier : NULL;
}

size_t ane_bridge_surface_alloc_bytes(ANEKernelHandle *kernel) {
    if (!kernel) return 0;
    return IOSurfaceGetAllocSize(kernel->ioInput);
}

void ane_bridge_free(ANEKernelHandle *kernel) {
    @autoreleasepool {
        if (!kernel) return;
        NSError *e = nil;
        if (kernel->model) {
            @try {
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                    model_obj(kernel), @selector(unloadWithQoS:error:), 21, &e);
            } @catch (NSException *ex) {
                // Best-effort unload; never throws out of free.
            }
        }
        if (kernel->request) CFBridgingRelease(kernel->request);
        if (kernel->model)   CFBridgingRelease(kernel->model);
        if (kernel->ioInput)  CFRelease(kernel->ioInput);
        if (kernel->ioOutput) CFRelease(kernel->ioOutput);
        // NOTE: temp dir is PRESERVED (compiler artifacts are inventoried by
        // the probe and cleaned by the controller only when authorized).
        free(kernel->tmpDir);
        free(kernel->identifier);
        free(kernel);
    }
}

int ane_bridge_get_compile_attempts(void) {
    return g_compile_attempts;
}

uint8_t *ane_bridge_build_weight_blob(const float *src, size_t n_elems, size_t *out_len) {
    // Byte layout copied verbatim from upstream ane_bridge_build_weight_blob
    // (identical to ane_mil_gen.h mil_build_weight_blob):
    //   128-byte header, then row-major fp16 data.
    if (!src || !out_len || n_elems == 0) return NULL;
    if (n_elems > (SIZE_MAX - 128) / 2) return NULL; // overflow guard
    size_t wsize = n_elems * 2; // fp16
    size_t total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    if (!buf) { if (out_len) *out_len = 0; return NULL; }

    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = (uint32_t)wsize;
    *(uint32_t*)(buf + 80) = 128;

    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (size_t i = 0; i < n_elems; i++) {
        fp16[i] = (_Float16)src[i];
    }

    *out_len = total;
    return buf;
}
