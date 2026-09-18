"""Generate and kernel-check a concrete acceptance proof for emitted bytes."""
from pathlib import Path
import hashlib
import subprocess
import time

from .encoding import command_dict, command_packet, nat

ROOT = Path(__file__).resolve().parents[2]


def lean_instruction(command):
    c = command_dict(command)
    kind = c['kind']
    def n(key):
        return str(nat(c[key]))
    if kind == 'config_ex':
        return '.configEx ' + n('dataflow')
    if kind == 'config_ld':
        if type(c['scale']) not in (int, float) or c['scale'] != 1.0:
            raise ValueError('unsupported scale')
        return '.configLd %s %s true' % (n('slot'), n('stride_bytes'))
    if kind == 'config_st':
        return '.configSt ' + n('stride_bytes')
    if kind == 'mvin':
        if c['buf'] not in ('A', 'B'):
            raise ValueError('unknown input')
        return '.mvin %s .%s %s %s %s %s' % (n('slot'), c['buf'].lower(), n('offset'), n('spad_addr'), n('cols'), n('rows'))
    if kind == 'preload':
        return '.preload %s %s' % (n('bd_spad_addr'), n('out_addr'))
    if kind == 'compute':
        if type(c['accumulated']) is not bool:
            raise ValueError('compute flag must be bool')
        return '.compute %s %s %s' % (str(c['accumulated']).lower(), n('a_spad_addr'), n('bd_spad_addr'))
    if kind == 'mvout':
        return '.mvout %s %s %s %s' % (n('buf_offset'), n('acc_addr'), n('cols'), n('rows'))
    return '.fence'


def certificate_source(plan, commands, encoding):
    if plan['schedule'] not in ('baseline', 'reuse_b'):
        raise ValueError('unsupported schedule')
    schedule = 'baseline' if plan['schedule'] == 'baseline' else 'reuseB'
    fields = [(name, nat(plan[key])) for name, key in (
        ('m', 'm'), ('n', 'n'), ('k', 'k'), ('dim', 'dim'),
        ('scratchpadRows', 'scratchpad_rows'), ('accumulatorRows', 'accumulator_rows'))]
    plan_lean = ', '.join('%s := %d' % item for item in fields) + ', schedule := .' + schedule
    bases = encoding['bases']
    base_lean = ', '.join('%s := %d' % (name.lower(), nat(bases[name])) for name in ('A', 'B', 'C'))
    code = bytes.fromhex(encoding['bytes_hex'])
    packets = [command_packet(c, bases) for c in commands if command_dict(c)['kind'] != 'fence']
    packet_lean = '[' + ','.join('⟨%d,%d,%d⟩' % packet for packet in packets) + ']'
    return '''import VeriTac.Gemmini.Executable
open VeriTac.Gemmini
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace SubmittedKernel
''' + 'def plan : GemminiPlan := { ' + plan_lean + ' }\n' + \
        'def bases : Encoding.Bases := { ' + base_lean + ' }\n' + \
        'def program : Program := [\n  ' + ',\n  '.join(lean_instruction(c) for c in commands) + '\n]\n' + \
        'def code : List UInt8 := [' + ','.join(map(str, code)) + ']\n' + \
        'def packets : List Encoding.OpPacket := ' + packet_lean + '\n' + '''
theorem symbolicAccepted : Symbolic.check plan program = true := by decide +kernel
theorem byteExecution : Encoding.executeBytes code = .ok packets := by decide +kernel
theorem commandDecoding : commandsOf bases packets = some program := by decide +kernel
theorem bufferLayout : Encoding.checkBases bases (bufferSizes plan) = true := by decide +kernel

theorem capacities :
    (plan.scratchpadRows ≤ 16384 && plan.accumulatorRows ≤ 1024) = true := by decide +kernel

theorem accepted : checkExecutable plan program bases code = true := by
  simp only [checkExecutable, capacities, symbolicAccepted, Bool.true_and,
    Encoding.decode, bufferLayout, if_true, byteExecution, commandDecoding]
  simp

theorem correct (a b : Nat → Int)
    (ha : ∀ i < plan.m * plan.k, VeriTac.GemminiExact.Int8 (a i))
    (hb : ∀ i < plan.k * plan.n, VeriTac.GemminiExact.Int8 (b i)) :
    Encoding.checkBases bases (bufferSizes plan) = true ∧
    ∃ s, runExecutable plan bases code a b = some s ∧ s.drained = true ∧
      ∀ r < plan.m, ∀ c < plan.n,
        s.output (r * plan.n + c) = gemm plan a b r c ∧ s.written (r * plan.n + c) = true :=
  checkExecutable_sound plan program bases code accepted a b ha hb

#print axioms accepted
#print axioms correct
end SubmittedKernel
'''


def write_and_check(plan, commands, encoding, output, timeout=300):
    """Save the actual body and certificate; accept only a successful Lean check."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / (plan['schedule'] + '_certificate.lean')
    binary = output / (plan['schedule'] + '.bin')
    log = output / (plan['schedule'] + '_certificate.log')
    code = bytes.fromhex(encoding['bytes_hex'])
    source.write_text(certificate_source(plan, commands, encoding))
    binary.write_bytes(code)
    started = time.monotonic()
    try:
        run = subprocess.run(['lake', 'env', 'lean', '-s', '65536', str(source)], cwd=ROOT,
                             capture_output=True, text=True, timeout=timeout)
        text = run.stdout + run.stderr
        accepted = run.returncode == 0 and "'SubmittedKernel.correct' depends on axioms:" in text \
            and 'sorryAx' not in text and 'Lean.ofReduceBool' not in text
        reason = 'kernel-checked concrete executable certificate' if accepted else 'Lean certificate rejected'
    except (OSError, subprocess.TimeoutExpired) as exc:
        accepted, reason, text = False, 'Lean certificate checker failed', str(exc)
    log.write_text(text)
    dependencies = {}
    for module in ('GemminiExact', 'Gemmini.Plan', 'Gemmini.Program', 'Gemmini.Program32',
                   'Gemmini.Word', 'Gemmini.Engine', 'Gemmini.EngineMap', 'Gemmini.Symbolic', 'Gemmini.SymbolicSound', 'Gemmini.Semantics', 'Gemmini.Execution16',
                   'Gemmini.Execution32', 'Gemmini.Checked', 'Gemmini.Encoding', 'Gemmini.Verification', 'Gemmini.Executable'):
        relative = 'VeriTac/' + module.replace('.', '/')
        for path in (ROOT / (relative + '.lean'), ROOT / '.lake/build/lib/lean' / (relative + '.olean')):
            if path.is_file():
                dependencies[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('lean-toolchain', 'lake-manifest.json'):
        path = ROOT / name
        if path.is_file():
            dependencies[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'accepted': accepted, 'reason': reason, 'proof': str(source),
            'binary': str(binary), 'log': str(log),
            'binary_sha256': hashlib.sha256(code).hexdigest(),
            'proof_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'dependency_sha256': dependencies, 'kernel_check_seconds': round(time.monotonic() - started, 3)}
