import sys,pathlib,json,ctypes as C,time
sys.path.insert(0,str(pathlib.Path('benchmarks/ane_e5_probe').resolve()))
import ane_e5_probe as p
out=pathlib.Path(sys.argv[2]).resolve();p.make_outdir(str(out));j=p.Journal(str(out/'journal.json'),{'experiment':'explicit load_for_execution; no compilation or evaluation','bundle':sys.argv[1]});created=[];rc=1
try:
 fns,err=p.load_functions('reopen',time.monotonic()+30,prepare=True)
 if err:raise p.ProbeError(err)
 dll=C.CDLL(p.FRAMEWORK);load=dll.e5rt_program_function_load_for_execution;load.restype=C.c_int64;load.argtypes=[p.P]
 lib=p.P();j.stage('library_create');r=fns['e5rt_program_library_create'](C.byref(lib),sys.argv[1].encode());p.check(r,lib,'library_create',j,float('inf'));created.append(('e5rt_program_library_release',lib))
 fn=p.P();j.stage('retain_main');r=fns['e5rt_program_library_retain_program_function'](lib,b'main',C.byref(fn));p.check(r,fn,'retain_main',j,float('inf'));created.append(('e5rt_program_function_release',fn))
 j.stage('load_for_execution');r=load(fn);p.check(r,None,'load_for_execution',j,float('inf'));rc=0
except Exception as e:j.fail(str(e));print(str(e),file=sys.stderr)
finally:
 if created:p.cleanup_all(fns,j,created)
 if rc==0:j.ok()
sys.exit(rc)
