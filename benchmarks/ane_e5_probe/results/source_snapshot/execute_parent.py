import pathlib,subprocess,sys,json,time
base=pathlib.Path('benchmarks/ane_e5_probe/results').resolve();mode,path,label=sys.argv[1:]
cmd=[sys.executable,str(pathlib.Path('benchmarks/ane_e5_probe/execute_control.py').resolve()),'--mode',mode,'--path',str(pathlib.Path(path).resolve()),'--input',str(pathlib.Path('benchmarks/ane/results/run-20260911-4/case-identity-c512s64/input.fp16.bin').resolve()),'--output',str(base/label),'--deadline','55'];t=time.monotonic()
with (base/(label+'.stdout.log')).open('w') as out,(base/(label+'.stderr.log')).open('w') as err:
 try:r=subprocess.run(cmd,stdout=out,stderr=err,timeout=60);rc=r.returncode
 except subprocess.TimeoutExpired:rc=124
(base/(label+'.process.json')).write_text(json.dumps({'command':cmd,'returncode':rc,'elapsed_s':time.monotonic()-t,'timeout_s':60},indent=2)+'\n');print('returncode',rc);print((base/(label+'.stderr.log')).read_text()[-5000:]);sys.exit(rc)
