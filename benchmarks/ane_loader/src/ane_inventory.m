#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <dlfcn.h>

static NSDictionary *fwVersion(NSString *path, void *dlres, const char *dlerr) {
  NSMutableDictionary *d = [@{@"path": path} mutableCopy];
  if (dlres != 0) {
    d[@"dlopen"] = @"ok";
    NSBundle *b = [NSBundle bundleWithPath:path];
    if (b) {
      d[@"bundlePresent"] = @YES;
      d[@"CFBundleVersion"] = b.infoDictionary[@"CFBundleVersion"] ?: @"";
      d[@"CFBundleShortVersionString"] = b.infoDictionary[@"CFBundleShortVersionString"] ?: @"";
    } else {
      d[@"bundlePresent"] = @NO;
    }
  } else {
    d[@"dlopen"] = @"failed";
    d[@"dlError"] = dlerr ? [NSString stringWithUTF8String:dlerr] : @"";
  }
  return d;
}

static NSArray *methodList(Class cls, BOOL meta) {
  unsigned int n = 0;
  Method *m = class_copyMethodList(meta ? object_getClass(cls) : cls, &n);
  NSMutableArray *out = [NSMutableArray array];
  for (unsigned int i = 0; i < n; i++) {
    [out addObject:@{
      @"name": [NSString stringWithUTF8String:sel_getName(method_getName(m[i]))],
      @"typeEncoding": [NSString stringWithUTF8String:method_getTypeEncoding(m[i])]}];
  }
  free(m);
  return [out sortedArrayUsingDescriptors:
    @[[NSSortDescriptor sortDescriptorWithKey:@"name" ascending:YES]]];
}

static NSArray *propertyList(Class cls) {
  unsigned int n = 0;
  objc_property_t *p = class_copyPropertyList(cls, &n);
  NSMutableArray *out = [NSMutableArray array];
  for (unsigned int i = 0; i < n; i++) {
    const char *an = property_getAttributes(p[i]);
    [out addObject:@{
      @"name": [NSString stringWithUTF8String:property_getName(p[i])],
      @"attributes": an ? [NSString stringWithUTF8String:an] : @""}];
  }
  free(p);
  return [out sortedArrayUsingDescriptors:
    @[[NSSortDescriptor sortDescriptorWithKey:@"name" ascending:YES]]];
}

static NSArray *protoMethodList(Protocol *p, BOOL required, BOOL instance) {
  unsigned int n = 0;
  struct objc_method_description *md = protocol_copyMethodDescriptionList(p, required, instance, &n);
  NSMutableArray *out = [NSMutableArray array];
  for (unsigned int i = 0; i < n; i++) {
    [out addObject:@{
      @"name": [NSString stringWithUTF8String:sel_getName(md[i].name)],
      @"typeEncoding": md[i].types ? [NSString stringWithUTF8String:md[i].types] : @""}];
  }
  free(md);
  return [out sortedArrayUsingDescriptors:
    @[[NSSortDescriptor sortDescriptorWithKey:@"name" ascending:YES]]];
}

static NSDictionary *invokeGetter(Class cls, NSString *selName, NSString *expEnc, NSArray *classMethods) {
  NSString *encoding = nil;
  for (NSDictionary *m in classMethods)
    if ([m[@"name"] isEqual:selName]) { encoding = m[@"typeEncoding"]; break; }
  if (!encoding)
    return @{@"selector": selName, @"status": @"method_not_found"};
  if (![encoding hasPrefix:expEnc])
    return @{@"selector": selName, @"status": @"type_mismatch",
             @"expectedEncodingPrefix": expEnc, @"actualEncoding": encoding};
  NSMutableDictionary *res = [@{@"selector": selName, @"encoding": encoding} mutableCopy];
  @try {
    SEL s = NSSelectorFromString(selName);
    NSMethodSignature *sig = [NSMethodSignature signatureWithObjCTypes:encoding.UTF8String];
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    [inv setTarget:(id)cls];
    [inv setSelector:s];
    [inv invoke];
    char r = [encoding UTF8String][0];
    if (r == '@') {
      __unsafe_unretained id v = nil;
      [inv getReturnValue:&v];
      res[@"value"] = v ?: @"";
      res[@"status"] = @"ok";
    } else if (r == 'B' || r == 'c') {
      BOOL b = NO;
      [inv getReturnValue:&b];
      res[@"value"] = @(b);
      res[@"status"] = @"ok";
    } else {
      res[@"status"] = @"unsupported_return";
    }
  } @catch (NSException *e) {
    res[@"status"] = @"exception";
    res[@"exception"] = [NSString stringWithFormat:@"%@: %@", e.name, e.reason];
  }
  return res;
}

int main(void) {
  @autoreleasepool {
    const char *f1 = "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
    const char *f2 = "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler";
    void *h1 = dlopen(f1, RTLD_NOW), *h2 = dlopen(f2, RTLD_NOW);
    const char *e1 = h1 ? NULL : dlerror();
    const char *e2 = h2 ? NULL : dlerror();

    unsigned int n = 0;
    Class *cs = objc_copyClassList(&n);
    NSMutableDictionary *classMap = [NSMutableDictionary dictionary];
    for (unsigned int i = 0; i < n; i++) {
      NSString *s = NSStringFromClass(cs[i]);
      if (![s containsString:@"ANE"]) continue;
      classMap[s] = @{
        @"superclass": cs[i] ? (class_getSuperclass(cs[i]) ? NSStringFromClass(class_getSuperclass(cs[i])) : @"") : @"",
        @"methods": methodList(cs[i], NO),
        @"classMethods": methodList(cs[i], YES),
        @"properties": propertyList(cs[i])};
    }
    free(cs);

    unsigned int pn = 0;
    Protocol *__unsafe_unretained *ps = objc_copyProtocolList(&pn);
    NSMutableDictionary *protoMap = [NSMutableDictionary dictionary];
    NSMutableArray *protocols = [NSMutableArray array];
    for (unsigned int i = 0; i < pn; i++) {
      NSString *name = [NSString stringWithUTF8String:protocol_getName(ps[i])];
      if (![name containsString:@"ANE"]) continue;
      [protocols addObject:name];
      protoMap[name] = @{
        @"requiredInstance": protoMethodList(ps[i], YES, YES),
        @"optionalInstance": protoMethodList(ps[i], NO, YES),
        @"requiredClass": protoMethodList(ps[i], YES, NO),
        @"optionalClass": protoMethodList(ps[i], NO, NO)};
    }
    free(ps);

    NSMutableArray *obs = [NSMutableArray array];
    Class deviceInfo = NSClassFromString(@"_ANEDeviceInfo");
    Class strings = NSClassFromString(@"_ANEStrings");
    NSArray *diMethods = classMap[@"_ANEDeviceInfo"][@"classMethods"];
    NSArray *stMethods = classMap[@"_ANEStrings"][@"classMethods"];
    if (deviceInfo) [obs addObject:invokeGetter(deviceInfo, @"precompiledModelChecksDisabled", @"B", diMethods)];
    if (strings) {
      [obs addObject:invokeGetter(strings, @"vm_allowPrecompiledBinaryBootArg", @"@", stMethods)];
      [obs addObject:invokeGetter(strings, @"testing_external_precompiledModelPath", @"@", stMethods)];
      [obs addObject:invokeGetter(strings, @"compilerServiceAccessEntitlement", @"@", stMethods)];
      [obs addObject:invokeGetter(strings, @"secondaryANECompilerServiceAccessEntitlement", @"@", stMethods)];
    }

    NSDictionary *json = @{
      @"probe": @"ane_loader/runtime-inventory",
      @"platform": @{@"os": [NSProcessInfo processInfo].operatingSystemVersionString},
      @"frameworks": @{
        @"AppleNeuralEngine": fwVersion(@"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework", h1, e1),
        @"ANECompiler": fwVersion(@"/System/Library/PrivateFrameworks/ANECompiler.framework", h2, e2)},
      @"classes": [[classMap allKeys] sortedArrayUsingSelector:@selector(compare:)],
      @"classDetail": classMap,
      @"protocols": [protocols sortedArrayUsingSelector:@selector(compare:)],
      @"protocolDetail": protoMap,
      @"runtime_observations": obs};

    NSData *data = [NSJSONSerialization dataWithJSONObject:json options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:nil];
    if (!data) { fprintf(stderr, "JSON serialization failed\n"); return 2; }
    NSString *out = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    NSString *dir = @"./results";
    if (![[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil]
        && ![[NSFileManager defaultManager] fileExistsAtPath:dir]) {
      fprintf(stderr, "failed to create results dir\n"); return 3;
    }
    NSString *dst = [dir stringByAppendingPathComponent:@"inventory.json"];
    NSError *werr = nil;
    if (![out writeToFile:dst atomically:YES encoding:NSUTF8StringEncoding error:&werr]) {
      fprintf(stderr, "write failed: %s\n", werr.localizedDescription.UTF8String ?: "?");
      return 4;
    }
    printf("%s\n", out.UTF8String);
  }
  return 0;
}
