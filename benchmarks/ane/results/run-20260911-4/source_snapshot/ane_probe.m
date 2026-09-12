// ane_probe.m — Bounded ANE feasibility probe CLI.
//
// Adapted ANE interaction lives in ane_bridge.m (see LICENSE-ANE.md).
// This file is the thin CLI: strict argument validation, MIL + weight-blob
// construction for a static FP16 1x1 conv + ReLU workload, and two explicit
// modes:
//   --mode metadata       read-only introspection (hw/os/toolchain/private
//                          API availability); never touches the ANE model.
//   --mode compile-and-run full pipeline (compile/load/dispatch) with
//                          compile and dispatch timed separately.
//
// The weight is a structured bounded matrix (identity or two-tap) so the
// controller can compute an independent float64 reference and verify forward
// orientation element-by-element.

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <mach/mach_time.h>
#import <sys/utsname.h>
#import <sys/sysctl.h>
#import <CommonCrypto/CommonDigest.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "ane_bridge.h"

#define PROBE_VERSION "0.1.0"
#define DEFAULT_CHANNELS 512
#define DEFAULT_SPATIAL  64
#define DEFAULT_QOS      21
#define DEFAULT_SAMPLES  3
#define DEFAULT_WARMUP   1
#define DEFAULT_SEED     1234

static mach_timebase_info_data_t g_tb;
static double ticks_to_ms(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
}

static NSString *sha256_hex(NSData *data) {
    if (!data) return @"";
    unsigned char md[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, md);
    NSMutableString *s = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH*2];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [s appendFormat:@"%02x", md[i]];
    return s;
}

static NSString *file_sha256(NSString *path) {
    NSData *d = [NSData dataWithContentsOfFile:path];
    return d ? sha256_hex(d) : @"";
}

// ---- MIL text: static FP16 1x1 conv (linear) + ReLU -----------------------
static NSString *build_mil(int C, int S) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> c = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = W, x = x)[name = string(\"conv\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> out = relu(x = c)[name = string(\"relu\")];\n"
        "    } -> (out);\n"
        "}\n",
        C, S, C, C, C, C, C, S, C, S];
}

// ---- Weight matrix, row-major [out, in]:
//      identity: W[o,o] = 1.0
//      two-tap : W[o,o] = 0.5, W[o,(o+1)%C] = -0.25 --------------------------
static float *build_weight(int C, const char *kind) {
    float *w = (float *)calloc((size_t)C * C, sizeof(float));
    if (!w) return NULL;
    float diag = (strcmp(kind, "identity") == 0) ? 1.0f : 0.5f;
    for (int o = 0; o < C; o++) {
        w[(size_t)o * C + o] = diag;
        if (strcmp(kind, "two-tap") == 0) {
            int in = (o + 1) % C;
            w[(size_t)o * C + in] = -0.25f;
        }
    }
    return w;
}

// ---- hardware / os / toolchain info ---------------------------------------
static NSDictionary *hw_info(void) {
    char model[256] = "", brand[256] = "";
    size_t model_len = sizeof(model), brand_len = sizeof(brand);
    sysctlbyname("hw.model", model, &model_len, NULL, 0);
    sysctlbyname("machdep.cpu.brand_string", brand, &brand_len, NULL, 0);
    uint64_t mem = 0; size_t mem_len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &mem_len, NULL, 0);
    int ncpu = 0; size_t ncpu_len = sizeof(ncpu);
    sysctlbyname("hw.ncpu", &ncpu, &ncpu_len, NULL, 0);
    struct utsname u; uname(&u);
    return @{
        @"hw_model": @(model),
        @"cpu_brand": @(brand),
        @"memsize_bytes": @(mem),
        @"logical_cpus": @(ncpu),
        @"machine": @(u.machine),
        @"nodename": @(u.nodename)
    };
}

static NSString *plist_version(NSString *path, NSString *key) {
    NSDictionary *d = [NSDictionary dictionaryWithContentsOfFile:path];
    id v = d ? d[key] : nil;
    return [v isKindOfClass:[NSString class]] ? v : @"";
}

static NSDictionary *toolchain_info(void) {
    return @{
        @"ANECompiler_CFBundleVersion": plist_version(
            @"/System/Library/PrivateFrameworks/ANECompiler.framework/Versions/A/Resources/Info.plist",
            @"CFBundleVersion"),
        @"CoreML_CFBundleVersion": plist_version(
            @"/System/Library/Frameworks/CoreML.framework/Versions/A/Resources/Info.plist",
            @"CFBundleVersion"),
        @"os": [[NSProcessInfo processInfo] operatingSystemVersionString]
    };
}

static NSDictionary *availability_to_dict(ANEBridgeAvailability *a) {
    return @{
        @"framework_loaded": @(a->framework_loaded),
        @"framework_path": @(a->framework_path),
        @"classes": @{
            @"_ANEClient": @(a->cls_client),
            @"_ANEInMemoryModelDescriptor": @(a->cls_descriptor),
            @"_ANEInMemoryModel": @(a->cls_inmem),
            @"_ANERequest": @(a->cls_request),
            @"_ANEIOSurfaceObject": @(a->cls_io)
        },
        @"methods": @{
            @"modelWithMILText:weights:optionsPlist:": @(a->m_modelWithMILText_weights_optionsPlist),
            @"inMemoryModelWithDescriptor:": @(a->m_inMemoryModelWithDescriptor),
            @"hexStringIdentifier": @(a->m_hexStringIdentifier),
            @"compileWithQoS:options:error:": @(a->m_compileWithQoS_options_error),
            @"loadWithQoS:options:error:": @(a->m_loadWithQoS_options_error),
            @"evaluateWithQoS:options:request:error:": @(a->m_evaluateWithQoS_options_request_error),
            @"unloadWithQoS:error:": @(a->m_unloadWithQoS_error),
            @"state": @(a->m_state)
        }
    };
}

// Recursively inventory a directory: relative path -> {size, sha256}.
// Skips directories and symlinks; hashes regular files only.
static NSDictionary *inventory_dir(NSString *root) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSArray *items = [fm subpathsOfDirectoryAtPath:root error:nil];
    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    for (NSString *rel in items) {
        if ([rel hasSuffix:@"/"]) continue;
        NSString *full = [root stringByAppendingPathComponent:rel];
        NSDictionary *attrs = [fm attributesOfItemAtPath:full error:nil];
        if (!attrs) continue;
        if ([attrs[NSFileType] isEqualToString:NSFileTypeDirectory]) continue;
        if ([attrs[NSFileType] isEqualToString:NSFileTypeSymbolicLink]) continue;
        if (![attrs[NSFileType] isEqualToString:NSFileTypeRegular]) continue;
        NSNumber *size = attrs[NSFileSize] ?: @0;
        out[rel] = @{ @"size_bytes": size, @"sha256": file_sha256(full) };
    }
    return out;
}

static void usage(void) {
    printf(
"Usage: ane_probe --mode <metadata|compile-and-run> [options]\n"
"\n"
"Modes:\n"
"  --mode metadata         read-only introspection only (no ANE model).\n"
"  --mode compile-and-run  compile, load, dispatch (timed separately).\n"
"\n"
"Options:\n"
"  --channels C        channels (256|512|1024)  [default 512]\n"
"  --spatial S         spatial  (16|64)         [default 64]\n"
"  --weight KIND       identity|two-tap          [default two-tap]\n"
"  --temp-policy P     isolated|system-model      [default isolated]\n"
"  --seed N            input seed (recorded; controller generates inputs)\n"
"  --qos N             compile/load/eval QoS     [default 21]\n"
"  --samples N         dispatch samples          [default 3]\n"
"  --warmup N          warmup evals              [default 1]\n"
"  --compile-budget N  <= %d\n"
"  --run-dir DIR       task-owned output dir (required)\n"
"  --input FILE        canonical fp16 input bytes (required for compile-and-run)\n"
"  --out FILE          results JSON path          [default run-dir/results.json]\n"
"  --help              this help\n",
    ANE_MAX_COMPILE_ATTEMPTS);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        NSString *mode = nil, *weightKind = @"two-tap", *runDir = nil, *inputFile = nil, *outFile = nil;
        NSString *tempPolicy = @"isolated";
        int C = DEFAULT_CHANNELS, S = DEFAULT_SPATIAL;
        int seed = DEFAULT_SEED, qos = DEFAULT_QOS;
        int samples = DEFAULT_SAMPLES, warmup = DEFAULT_WARMUP, budget = ANE_MAX_COMPILE_ATTEMPTS;
        bool haveBudget = false;

        for (int i = 1; i < argc; i++) {
            NSString *arg = [NSString stringWithUTF8String:argv[i]];
            if ([arg isEqualToString:@"--help"]) { usage(); return 0; }
            if (i + 1 >= argc) { fprintf(stderr, "ane_probe: missing value for %s\n", argv[i]); return 2; }
            NSString *val = [NSString stringWithUTF8String:argv[++i]];
            if ([arg isEqualToString:@"--mode"]) mode = val;
            else if ([arg isEqualToString:@"--channels"]) C = val.intValue;
            else if ([arg isEqualToString:@"--spatial"]) S = val.intValue;
            else if ([arg isEqualToString:@"--weight"]) weightKind = val;
            else if ([arg isEqualToString:@"--temp-policy"]) tempPolicy = val;
            else if ([arg isEqualToString:@"--seed"]) seed = val.intValue;
            else if ([arg isEqualToString:@"--qos"]) qos = val.intValue;            else if ([arg isEqualToString:@"--samples"]) samples = val.intValue;
            else if ([arg isEqualToString:@"--warmup"]) warmup = val.intValue;
            else if ([arg isEqualToString:@"--compile-budget"]) { budget = val.intValue; haveBudget = true; }
            else if ([arg isEqualToString:@"--run-dir"]) runDir = val;
            else if ([arg isEqualToString:@"--input"]) inputFile = val;
            else if ([arg isEqualToString:@"--out"]) outFile = val;
            else { fprintf(stderr, "ane_probe: unknown option %s\n", argv[i]); return 2; }
        }

        // Strict validation.
        if (!mode) { fprintf(stderr, "ane_probe: --mode is required\n"); return 2; }
        if (![mode isEqualToString:@"metadata"] && ![mode isEqualToString:@"compile-and-run"]) {
            fprintf(stderr, "ane_probe: invalid --mode '%s'\n", mode.UTF8String); return 2;
        }
        if (!runDir) { fprintf(stderr, "ane_probe: --run-dir is required\n"); return 2; }
        if (C != 256 && C != 512 && C != 1024) { fprintf(stderr, "ane_probe: invalid channels\n"); return 2; }
        if (S != 16 && S != 64) { fprintf(stderr, "ane_probe: invalid spatial\n"); return 2; }
        if (![weightKind isEqualToString:@"identity"] && ![weightKind isEqualToString:@"two-tap"]) {
            fprintf(stderr, "ane_probe: invalid --weight\n"); return 2; }
        if (![tempPolicy isEqualToString:@"isolated"] &&
            ![tempPolicy isEqualToString:@"system-model"]) {
            fprintf(stderr, "ane_probe: invalid --temp-policy (isolated|system-model)\n"); return 2; }
        if (seed < 0 || samples < 1 || samples > 100 || warmup < 0 || warmup > 10) {
            fprintf(stderr, "ane_probe: bad numeric arg (samples in [1,100], warmup in [0,10])\n"); return 2; }
        if (qos != 21) { fprintf(stderr, "ane_probe: only --qos 21 is supported\n"); return 2; }
        if (haveBudget && (budget < 1 || budget > ANE_MAX_COMPILE_ATTEMPTS)) {
            fprintf(stderr, "ane_probe: --compile-budget must be in [1,%d]\n", ANE_MAX_COMPILE_ATTEMPTS); return 2; }
        if ([mode isEqualToString:@"compile-and-run"] && !inputFile) {
            fprintf(stderr, "ane_probe: --input required for compile-and-run\n"); return 2; }

        NSFileManager *fm = [NSFileManager defaultManager];
        NSError *mkdirErr = nil;
        if (![fm createDirectoryAtPath:runDir withIntermediateDirectories:YES attributes:nil error:&mkdirErr]) {
            fprintf(stderr, "ane_probe: cannot create run-dir: %s\n", mkdirErr.description.UTF8String); return 1;
        }
        if (!outFile) outFile = [runDir stringByAppendingPathComponent:@"results.json"];

        // ---- build workload (deterministic, no ANE) -----------------------
        NSString *mil = build_mil(C, S);
        float *w = build_weight(C, weightKind.UTF8String);
        if (!w) { fprintf(stderr, "ane_probe: weight alloc failed\n"); return 1; }
        size_t blobLen = 0;
        uint8_t *blob = ane_bridge_build_weight_blob(w, (size_t)C * C, &blobLen);
        free(w);
        if (!blob) { fprintf(stderr, "ane_probe: blob build failed\n"); return 1; }

        size_t inBytes = (size_t)C * S * 2;   // fp16
        size_t outBytes = inBytes;
        size_t strideBytes = (inBytes + 15) & ~(size_t)15; // 16-byte aligned padding

        // Write the probe's independently generated MIL + blob for hash
        // cross-check by the controller.
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        NSData *blobData = [NSData dataWithBytes:blob length:blobLen];
        NSString *milPath = [runDir stringByAppendingPathComponent:@"model.mil"];
        NSString *blobPath = [runDir stringByAppendingPathComponent:@"weight.bin"];
        if (![milData writeToFile:milPath atomically:YES] ||
            ![blobData writeToFile:blobPath atomically:YES]) {
            fprintf(stderr, "ane_probe: could not write model.mil / weight.bin\n");
            free(blob);
            return 1;
        }

        // ---- availability (read-only) --------------------------------------
        ANEBridgeAvailability avail;
        int initrc = ane_bridge_init(&avail);

        NSMutableDictionary *res = [NSMutableDictionary dictionary];
        res[@"probe_version"] = @PROBE_VERSION;
        res[@"mode"] = mode;
        res[@"hw"] = hw_info();
        res[@"toolchain"] = toolchain_info();
        res[@"private_api"] = availability_to_dict(&avail);
        res[@"workload"] = @{
            @"kind": @"conv1x1_relu",
            @"dtype": @"fp16",
            @"channels": @(C),
            @"spatial": @(S),
            @"weight": @{ @"kind": weightKind,
                          @"diag": [weightKind isEqualToString:@"identity"] ? @1.0 : @0.5,
                          @"offdiag": [weightKind isEqualToString:@"two-tap"] ? @(-0.25) : @0 },
            @"seed": @(seed),
            @"qos": @(qos),
            @"input_shape": @[@1, @(C), @1, @(S)],
            @"weight_shape": @[@(C), @(C), @1, @1],
            @"output_shape": @[@1, @(C), @1, @(S)],
            @"mil_text": mil,
            @"layout": @{
                @"input_bytes": @(inBytes),
                @"output_bytes": @(outBytes),
                @"io_stride_pad_bytes": @(strideBytes),
                @"pad_align_bytes": @16,
                @"iosurface": @{ @"width_bytes": @(strideBytes), @"height": @1,
                                 @"bytes_per_element": @1, @"bytes_per_row": @(strideBytes),
                                 @"alloc_size": @(strideBytes), @"pixel_format": @0 }
            }
        };
        NSMutableDictionary *hashes = [NSMutableDictionary dictionary];
        hashes[@"mil_sha256"] = sha256_hex(milData);
        hashes[@"weight_blob_sha256"] = sha256_hex(blobData);
        res[@"hashes"] = hashes;
        // Source/binary hashes are computed by the controller (it owns the
        // source paths); the probe has no stable self-path to hash.

        if ([mode isEqualToString:@"metadata"]) {
            NSData *json = [NSJSONSerialization dataWithJSONObject:res
                                options:NSJSONWritingPrettyPrinted error:nil];
            if (!json || ![json writeToFile:outFile atomically:YES]) {
                fprintf(stderr, "ane_probe: could not write metadata JSON\n");
                free(blob);
                return 1;
            }
            printf("ane_probe: metadata mode complete\n");
            printf("  private api init rc=%d, framework_loaded=%d\n",
                   initrc, avail.framework_loaded ? 1 : 0);
            printf("  classes: client=%d desc=%d inmem=%d req=%d io=%d\n",
                   avail.cls_client?1:0, avail.cls_descriptor?1:0, avail.cls_inmem?1:0,
                   avail.cls_request?1:0, avail.cls_io?1:0);
            printf("  results: %s\n", outFile.UTF8String);
            free(blob);
            return initrc == 0 ? 0 : 1;
        }

        // ---- compile-and-run (phase-2 authorized; worker runs 2 cases) ----
        int exitCode = 0;
        ANEKernelHandle *k = NULL;
        char err[2048] = {0};
        char err2[512] = {0};

        NSMutableDictionary *compile   = [NSMutableDictionary dictionary];
        NSMutableDictionary *load      = [NSMutableDictionary dictionary];
        NSMutableDictionary *dispatch  = [NSMutableDictionary dictionary];
        NSMutableDictionary *transfer  = [NSMutableDictionary dictionary];
        NSMutableDictionary *output    = [NSMutableDictionary dictionary];
        res[@"compile"] = compile;
        res[@"load"] = load;
        res[@"dispatch"] = dispatch;
        res[@"transfer"] = transfer;
        res[@"output"] = output;

        {
        // Preflight: temp isolation must hold before ANY ANE work.
        //   isolated     : NSTemporaryDirectory() must resolve to
        //                  runDir/framework_tmp (task-owned).
        //   system-model : NSTemporaryDirectory() must be a nonempty absolute
        //                  directory (Foundation actual temp root); no env
        //                  spoofing, no security change. The bridge still
        //                  exclusively creates the owned model subdir.
        NSMutableDictionary *preflight = [NSMutableDictionary dictionary];
        res[@"preflight"] = preflight;
        res[@"temp_policy"] = tempPolicy;
        NSString *actualTmp = [NSTemporaryDirectory() stringByStandardizingPath];
        NSString *envTmp = [[[NSProcessInfo processInfo] environment] objectForKey:@"TMPDIR"] ?: @"";
        preflight[@"policy"] = tempPolicy;
        preflight[@"actual_tmp"] = actualTmp;
        preflight[@"env_TMPDIR"] = envTmp;
        BOOL tmpOk = NO;
        if ([tempPolicy isEqualToString:@"isolated"]) {
            NSString *expectedTmp = [[runDir stringByAppendingPathComponent:@"framework_tmp"]
                stringByStandardizingPath];
            preflight[@"expected_tmp"] = expectedTmp;
            tmpOk = [actualTmp isEqualToString:expectedTmp] &&
                    envTmp.length > 0 &&
                    [envTmp isEqualToString:expectedTmp];
        } else { // system-model
            BOOL absolute = [actualTmp hasPrefix:@"/"];
            BOOL isDir = NO;
            [[NSFileManager defaultManager] fileExistsAtPath:actualTmp isDirectory:&isDir];
            preflight[@"actual_is_absolute"] = @(absolute);
            preflight[@"actual_is_dir"] = @(isDir);
            tmpOk = actualTmp.length > 0 && absolute && isDir;
        }
preflight[@"temp_policy_ok"] = @(tmpOk);
preflight[@"tmp_isolated_ok"] = @([tempPolicy isEqualToString:@"isolated"] && tmpOk);
        if (!tmpOk) {
            fprintf(stderr,
                "ane_probe: temp preflight FAILED (policy=%s)\n  actual=%s\n  env   =%s\n",
                tempPolicy.UTF8String, actualTmp.UTF8String, envTmp.UTF8String);
            exitCode = 1;
            goto done;
        }

        NSData *input = [NSData dataWithContentsOfFile:inputFile];
        if (!input || input.length != inBytes) {
            fprintf(stderr, "ane_probe: input file missing or size != %zu\n", inBytes);
            exitCode = 1;
            goto done;
        }
        hashes[@"input_sha256"] = sha256_hex(input);
        res[@"workload_input_bytes"] = @(input.length);

        // --- compile (timed separately) ---
        uint64_t t0 = mach_absolute_time();
        k = ane_bridge_compile(mil.UTF8String, blob, blobLen,
                               inBytes, outBytes, strideBytes, err, sizeof(err));
        uint64_t t1 = mach_absolute_time();
        compile[@"elapsed_ms"] = @(ticks_to_ms(t1 - t0));
        compile[@"compile_attempts_after"] = @(ane_bridge_get_compile_attempts());
        res[@"temp_dir_root"] = [NSTemporaryDirectory() stringByStandardizingPath];
        if (!k) {
            compile[@"status"] = @"failed";
            compile[@"error"] = @(err);
            fprintf(stderr, "ane_probe: compile failed: %s\n", err);
            exitCode = 1; // stop hardware work; preserve artifacts
            goto done;
        }
        compile[@"status"] = @"ok";
        compile[@"model_identifier"] = @(ane_bridge_identifier(k));
        compile[@"owned_model_tmpdir"] = ane_bridge_last_tmpdir() ? @(ane_bridge_last_tmpdir()) : @"";
        compile[@"surface_alloc_bytes"] = @(ane_bridge_surface_alloc_bytes(k));

        // --- load (timed separately) ---
        err[0] = 0;
        t0 = mach_absolute_time();
        int lrc = ane_bridge_load(k, err, sizeof(err));
        t1 = mach_absolute_time();
        load[@"elapsed_ms"] = @(ticks_to_ms(t1 - t0));
        load[@"status"] = (lrc == 0) ? @"ok" : @"failed";
        if (lrc != 0) {
            load[@"error"] = @(err);
            fprintf(stderr, "ane_probe: load failed: %s\n", err);
            exitCode = 1; // stop hardware work; preserve artifacts
            goto done;
        }
        load[@"state_after"] = @(ane_bridge_state(k));

        // --- transfer input (CPU -> IOSurface) ---
        t0 = mach_absolute_time();
        int wrc = ane_bridge_write_input(k, input.bytes, input.length);
        t1 = mach_absolute_time();
        transfer[@"write_input_ms"] = @(ticks_to_ms(t1 - t0));
        if (wrc != 0) {
            transfer[@"write_input_status"] = @"failed";
            fprintf(stderr, "ane_probe: write_input failed\n");
            exitCode = 1;
            goto done;
        }
        transfer[@"write_input_status"] = @"ok";

        // --- warmup + dispatch samples (each timed) ---
        bool dispatchOK = true;
        for (int i = 0; i < warmup; i++) {
            err2[0] = 0;
            if (ane_bridge_eval(k, err2, sizeof(err2)) != 0) {
                dispatch[@"status"] = @"failed";
                dispatch[@"error"] = @(err2);
                fprintf(stderr, "ane_probe: warmup eval failed: %s\n", err2);
                dispatchOK = false;
                exitCode = 1;
                goto done;
            }
        }
        NSMutableArray *perEval = [NSMutableArray array];
        for (int i = 0; i < samples; i++) {
            err2[0] = 0;
            t0 = mach_absolute_time();
            int erc = ane_bridge_eval(k, err2, sizeof(err2));
            t1 = mach_absolute_time();
            [perEval addObject:@(ticks_to_ms(t1 - t0))];
            if (erc != 0) {
                dispatchOK = false;
                dispatch[@"status"] = @"failed";
                dispatch[@"error"] = @(err2);
                dispatch[@"failed_sample_index"] = @(i);
                fprintf(stderr, "ane_probe: sample %d eval failed: %s\n", i, err2);
                exitCode = 1;
                break; // do NOT read stale output
            }
        }
        dispatch[@"per_eval_elapsed_ms"] = perEval;
        dispatch[@"samples"] = @(samples);
        dispatch[@"warmup"] = @(warmup);
        if (dispatchOK) {
            dispatch[@"status"] = @"ok";
            double sum = 0; for (NSNumber *n in perEval) sum += n.doubleValue;
            if (perEval.count > 0) dispatch[@"mean_elapsed_ms"] = @(sum / perEval.count);
        }

        if (!dispatchOK) goto done; // no stale output read

        // --- read output (IOSurface -> CPU) + hash + write ---
        NSMutableData *out = [NSMutableData dataWithLength:outBytes];
        t0 = mach_absolute_time();
        int rrc = ane_bridge_read_output(k, out.mutableBytes, out.length);
        t1 = mach_absolute_time();
        transfer[@"read_output_ms"] = @(ticks_to_ms(t1 - t0));
        transfer[@"read_output_status"] = (rrc == 0) ? @"ok" : @"failed";
        if (rrc != 0) {
            fprintf(stderr, "ane_probe: read_output failed\n");
            exitCode = 1;
            goto done;
        }
        NSString *outPath = [runDir stringByAppendingPathComponent:@"output.fp16.bin"];
        if (![out writeToFile:outPath atomically:YES]) {
            fprintf(stderr, "ane_probe: could not write output file\n");
            exitCode = 1;
            goto done;
        }
        hashes[@"output_sha256"] = sha256_hex(out);
        output[@"path"] = outPath;
        output[@"bytes"] = @(out.length);
        output[@"sha256"] = sha256_hex(out);
        }

done: ;
        // --- common checked finalization: preserve artifacts + results JSON ---
        // Inventory and copy ONLY the bridge-owned model temp dir
        // (ane_bridge_last_tmpdir), never the NSTemporaryDirectory() parent.
        const char *owned = ane_bridge_last_tmpdir();
        if (owned && owned[0]) {
            NSString *ownedStr = @(owned);
            res[@"owned_model_tmpdir"] = ownedStr;
            res[@"artifacts_compiler"] = inventory_dir(ownedStr);
            NSString *dest = [runDir stringByAppendingPathComponent:@"compiler_artifacts"];
            res[@"compiler_artifacts_dir"] = dest;
            BOOL destExists = [fm fileExistsAtPath:dest];
            res[@"compiler_artifacts_overwrite_rejected"] = @(destExists);
            if (!destExists) {
                NSError *copyErr = nil;
                BOOL copied = [fm copyItemAtPath:ownedStr toPath:dest error:&copyErr];
                res[@"compiler_artifacts_copied"] = @(copied);
                if (!copied) {
                    res[@"compiler_artifacts_copy_error"] = copyErr.description ?: @"";
                    exitCode = 1;
                }
            } else {
                res[@"compiler_artifacts_copied"] = @NO;
                exitCode = 1;
            }
        }
        if (k) {
            ane_bridge_free(k);
            k = NULL;
        }
        free(blob);
        blob = NULL;
        res[@"exit_code"] = @(exitCode);
        NSData *json = [NSJSONSerialization dataWithJSONObject:res
                            options:NSJSONWritingPrettyPrinted error:nil];
        if (!json) {
            fprintf(stderr, "ane_probe: JSON serialization failed\n");
            return 1;
        }
        if (![json writeToFile:outFile atomically:YES]) {
            fprintf(stderr, "ane_probe: could not write results JSON\n");
            return 1;
        }
        printf("ane_probe: compile-and-run finished (exit %d) -> %s\n", exitCode, outFile.UTF8String);
        return exitCode;
    }
}
