import VeriTac.Tenstorrent.GraphCertificate

/-! Restricted Blackhole NoC/CB protocol. One issuer, immutable input pages,
disjoint input/output regions and CB slots, asynchronous atomic page completion.
Physical DMA progress and the Metalium implementation are explicit assumptions.
No C++, instruction encoding, floating-point or silicon theorem is claimed. -/
namespace VeriTac.Tenstorrent

structure Task where
  inputPages : Nat
  expected : List Nat
  slots : Nat
  pageBytes : Nat
  deriving DecidableEq, Repr

inductive Instr where
  | reserve (slot : Nat)
  | read (slot src : Nat)
  | waitReads
  | publish (slot : Nat)
  | acquire (slot : Nat)
  | write (slot dst : Nat)
  | waitWrites
  | release (slot : Nat)
  deriving DecidableEq, Repr

inductive Phase where
  | free | reserved | published | acquired
  deriving DecidableEq, Repr

structure State where
  pc : Nat
  phases : List Phase
  values : List (Option Nat)
  outputs : List (Option Nat)
  reads : List (Nat × Nat)
  writes : List (Nat × Nat × Nat)
  deriving DecidableEq, Repr

def initial (t : Task) : State :=
  ⟨0, List.replicate t.slots .free, List.replicate t.slots none,
   List.replicate t.expected.length none, [], []⟩

def validTask (t : Task) : Bool :=
  decide (0 < t.inputPages ∧ t.inputPages ≤ 16 ∧
          0 < t.expected.length ∧ t.expected.length ≤ 16 ∧
          0 < t.slots ∧ t.slots ≤ 8 ∧
          32 ≤ t.pageBytes ∧ t.pageBytes ≤ 16384 ∧ t.pageBytes % 32 = 0) &&
  t.expected.all (fun x => decide (x < t.inputPages))

def validState (t : Task) (s : State) : Bool :=
  decide (s.phases.length = t.slots ∧ s.values.length = t.slots ∧
          s.outputs.length = t.expected.length) &&
  s.reads.all (fun x => decide (x.1 < t.slots ∧ x.2 < t.inputPages)) &&
  s.writes.all (fun x => decide (x.1 < t.slots ∧ x.2.1 < t.expected.length ∧ x.2.2 < t.inputPages))

def valueAt (s : State) (slot : Nat) : Option Nat :=
  match s.values[slot]? with | some v => v | none => none

def phaseAt (s : State) (slot : Nat) (p : Phase) : Bool :=
  decide (s.phases[slot]? = some p)

def pendingRead (s : State) (slot : Nat) : Bool := s.reads.any (fun r => r.1 == slot)
def pendingWrite (s : State) (slot : Nat) : Bool := s.writes.any (fun w => w.1 == slot)
def pendingDest (s : State) (dst : Nat) : Bool := s.writes.any (fun w => w.2.1 == dst)

def readLE (a b : Nat × Nat) : Bool :=
  decide (a.1 < b.1) || (a.1 == b.1 && decide (a.2 ≤ b.2))
def writeLE (a b : Nat × Nat × Nat) : Bool :=
  decide (a.1 < b.1) || (a.1 == b.1 && readLE a.2 b.2)

/-- Structurally recursive insertion keeps kernel reduction transparent. -/
def insertRead (x : Nat × Nat) : List (Nat × Nat) → List (Nat × Nat)
  | [] => [x]
  | y :: ys => if readLE x y then x :: y :: ys else y :: insertRead x ys

def insertWrite (x : Nat × Nat × Nat) : List (Nat × Nat × Nat) → List (Nat × Nat × Nat)
  | [] => [x]
  | y :: ys => if writeLE x y then x :: y :: ys else y :: insertWrite x ys

/-- none is a blocked barrier or end of program; error is an unsafe issue.
Completion is deliberately not assumed to happen before an unsafe issue. -/
def issue (t : Task) (program : List Instr) (s : State) : Except String (Option State) := do
  match program[s.pc]? with
  | none => return none
  | some op =>
    let advanced := {s with pc := s.pc + 1}
    match op with
    | .waitReads => return if s.reads.isEmpty then some advanced else none
    | .waitWrites => return if s.writes.isEmpty then some advanced else none
    | .reserve slot =>
      if slot < t.slots && phaseAt s slot .free then
        return some {advanced with phases := s.phases.set slot .reserved, values := s.values.set slot none}
      else throw "reserve requires a free in-bounds slot"
    | .read slot src =>
      if slot < t.slots && src < t.inputPages && phaseAt s slot .reserved &&
          !(pendingRead s slot) && !(pendingWrite s slot) then
        return some {advanced with values := s.values.set slot none, reads := insertRead (slot, src) s.reads}
      else throw "read requires reserved non-busy slot and in-bounds source"
    | .publish slot =>
      if slot < t.slots && phaseAt s slot .reserved && (valueAt s slot).isSome && !(pendingRead s slot) then
        return some {advanced with phases := s.phases.set slot .published}
      else throw "publish requires a completed initialized read"
    | .acquire slot =>
      if slot < t.slots && phaseAt s slot .published then
        return some {advanced with phases := s.phases.set slot .acquired}
      else throw "acquire requires published slot"
    | .write slot dst =>
      if slot < t.slots && dst < t.expected.length && phaseAt s slot .acquired &&
          !(pendingRead s slot) && !(pendingDest s dst) then
        match valueAt s slot with
        | none => throw "write requires initialized value"
        | some origin => return some {advanced with
            writes := insertWrite (slot, dst, origin) s.writes}
      else throw "write requires acquired non-busy source and non-busy destination"
    | .release slot =>
      if slot < t.slots && phaseAt s slot .acquired && !(pendingRead s slot) && !(pendingWrite s slot) then
        return some {advanced with phases := s.phases.set slot .free, values := s.values.set slot none}
      else throw "release requires acquired slot with no pending transfer"

/-- Reads and writes may complete in any order. The index identifies an event,
not a data value, so nondeterministic completion never assumes data equality. -/
def completeRead (s : State) (i : Nat) : Option State := do
  let (slot, src) ← s.reads[i]?
  return {s with reads := s.reads.eraseIdx i, values := s.values.set slot (some src)}

def completeWrite (s : State) (i : Nat) : Option State := do
  let (_, dst, origin) ← s.writes[i]?
  return {s with writes := s.writes.eraseIdx i, outputs := s.outputs.set dst (some origin)}

def successors (t : Task) (program : List Instr) (s : State) : List State :=
  (match issue t program s with | .ok (some q) => [q] | _ => []) ++
  (List.range s.reads.length).filterMap (completeRead s) ++
  (List.range s.writes.length).filterMap (completeWrite s)

def issueSafe (t : Task) (program : List Instr) (s : State) : Bool :=
  match issue t program s with | .error _ => false | .ok _ => true

def finalGood (t : Task) (program : List Instr) (s : State) : Bool :=
  decide (s.pc = program.length ∧ s.reads = [] ∧ s.writes = [] ∧
          s.phases = List.replicate t.slots .free ∧ s.outputs = t.expected.map some)

/-- Every genuine issue/completion event must strictly decrease this rank.
Idle/stuttering time is not an event; hardware fairness remains an assumption. -/
def potential (program : List Instr) (s : State) : Nat :=
  2 * (program.length - s.pc) + s.reads.length + s.writes.length

def decreases (t : Task) (program : List Instr) (s : State) : Bool :=
  (successors t program s).all (fun q => decide (potential program q < potential program s))

def localGood (t : Task) (program : List Instr) (s : State) : Bool :=
  validState t s && issueSafe t program s && decreases t program s &&
  (if (successors t program s).isEmpty then finalGood t program s else true)

/-- A proposed graph is untrusted data. Lean checks its initial state, every
node and ALL modeled successors; extra safe nodes are harmless. -/
def checkCertificate (t : Task) (program : List Instr) (nodes : List State) : Bool :=
  validTask t && decide (0 < program.length ∧ program.length ≤ 128) &&
  ProtocolGraph.check (initial t) (successors t program) (localGood t program) nodes

abbrev Reachable (t : Task) (program : List Instr) :=
  ProtocolGraph.Reach (initial t) (successors t program)

theorem accepted_localGood (t : Task) (program : List Instr) (nodes : List State)
    (h : checkCertificate t program nodes = true) :
    ∀ s, Reachable t program s → localGood t program s = true := by
  have hg : ProtocolGraph.check (initial t) (successors t program) (localGood t program) nodes = true :=
    (Bool.and_eq_true_iff.mp h).2
  exact ProtocolGraph.check_sound _ _ _ _ hg

theorem accepted_issueSafe (t : Task) (program : List Instr) (nodes : List State)
    (h : checkCertificate t program nodes = true) (s : State) (hr : Reachable t program s) :
    issueSafe t program s = true := by
  have hg := accepted_localGood t program nodes h s hr
  exact (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp hg).1).1).2

theorem accepted_terminal (t : Task) (program : List Instr) (nodes : List State)
    (h : checkCertificate t program nodes = true) (s : State) (hr : Reachable t program s)
    (ht : successors t program s = []) : finalGood t program s = true := by
  have hg := (Bool.and_eq_true_iff.mp (accepted_localGood t program nodes h s hr)).2
  simpa [ht] using hg

theorem accepted_decreases (t : Task) (program : List Instr) (nodes : List State)
    (h : checkCertificate t program nodes = true) (s : State) (hr : Reachable t program s)
    (q : State) (hq : q ∈ successors t program s) :
    potential program q < potential program s := by
  have hd := (Bool.and_eq_true_iff.mp (Bool.and_eq_true_iff.mp
    (accepted_localGood t program nodes h s hr)).1).2
  simp only [decreases, List.all_eq_true, decide_eq_true_eq] at hd
  exact hd q hq

/-- Every modeled event path is well founded. This rules out an infinite
sequence of protocol events, not an indefinitely stalled physical DMA. -/
theorem accepted_accessible (t : Task) (program : List Instr) (nodes : List State)
    (h : checkCertificate t program nodes = true) (s : State) (hr : Reachable t program s) :
    Acc (fun q s => q ∈ successors t program s) s := by
  have aux : ∀ n s, Reachable t program s → potential program s = n →
      Acc (fun q s => q ∈ successors t program s) s := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro s hs heq
      apply Acc.intro
      intro q hq
      have hd := accepted_decreases t program nodes h s hs q hq
      exact ih (potential program q) (by simpa [heq] using hd) q
        (ProtocolGraph.Reach.step hs hq) rfl
  exact aux _ s hr rfl

/-- Contents semantics: the input pages are immutable and the only data
operation is exact copy, so each initialized page denotes its source payload.
This interpretation does not assume injective/distinct input payloads. -/
def outputPayloads {α : Type} (input : Nat → α) (s : State) : List (Option α) :=
  s.outputs.map (Option.map input)

theorem accepted_output_all_payloads (t : Task) (program : List Instr) (nodes : List State)
    (h : checkCertificate t program nodes = true) (s : State) (hr : Reachable t program s)
    (ht : successors t program s = []) {α : Type} (input : Nat → α) :
    outputPayloads input s = t.expected.map (fun i => some (input i)) := by
  have hf := accepted_terminal t program nodes h s hr ht
  have hout : s.outputs = t.expected.map some := by
    simp only [finalGood, decide_eq_true_eq] at hf
    exact hf.2.2.2.2
  simp [outputPayloads, hout, List.map_map]

#print axioms accepted_accessible
#print axioms accepted_issueSafe
#print axioms accepted_output_all_payloads
end VeriTac.Tenstorrent
