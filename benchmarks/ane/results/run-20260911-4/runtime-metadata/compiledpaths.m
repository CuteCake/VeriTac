#import <Foundation/Foundation.h>
#import <objc/message.h>
#include "/Users/cake/.codex/worktrees/bfe0/VeriTac/benchmarks/ane/src/ane_bridge.m"
int main(int argc,const char**argv){@autoreleasepool{
 if(argc!=3)return 2;ane_bridge_init(NULL);
 NSString*mil=[[NSString alloc]initWithData:[NSData dataWithContentsOfFile:@(argv[1])] encoding:NSUTF8StringEncoding];mil=[mil stringByAppendingString:@"\n\n"];
 NSData*w=[NSData dataWithContentsOfFile:@(argv[2])];char err[2048]={0};
 ANEKernelHandle*k=ane_bridge_compile(mil.UTF8String,w.bytes,w.length,65536,65536,65536,err,sizeof err);
 printf("compile_ok=%d error=%s owned=%s\n",k!=NULL,err,ane_bridge_last_tmpdir()?:"");if(!k)return 1;
 id mdl=model_obj(k);
 for(NSString*key in @[@"modelURL",@"localModelPath",@"modelAttributes",@"model"]){SEL s=NSSelectorFromString(key);if(![mdl respondsToSelector:s])continue;id v=((id(*)(id,SEL))objc_msgSend)(mdl,s);NSString*d=[v description]?:@"nil";printf("%s=%s\n",key.UTF8String,[[d substringToIndex:MIN(d.length,8192)]UTF8String]);
 if([key isEqualToString:@"model"]&&v){for(NSString*key2 in @[@"modelURL",@"sourceURL",@"cacheURLIdentifier",@"modelAttributes"]){SEL s2=NSSelectorFromString(key2);if(![v respondsToSelector:s2])continue;id v2=((id(*)(id,SEL))objc_msgSend)(v,s2);NSString*d2=[v2 description]?:@"nil";printf("inner.%s=%s\n",key2.UTF8String,[[d2 substringToIndex:MIN(d2.length,8192)]UTF8String]);}}
 }
 ane_bridge_free(k);return 0;
}}
