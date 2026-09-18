import VeriTac.Tenstorrent.Protocol
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace SubmittedProtocol
open VeriTac.Tenstorrent
def task : Task := ⟨2, [0, 1], 2, 512⟩
def program : List Instr := [.reserve 0, .reserve 1, .read 0 0, .read 1 1, .waitReads, .publish 0, .publish 1, .acquire 0, .acquire 1, .write 0 0, .write 1 1, .waitWrites, .release 0, .release 1]
def nodes : List State := [
⟨0, [.free, .free], [none, none], [none, none], [], []⟩,
⟨1, [.reserved, .free], [none, none], [none, none], [], []⟩,
⟨2, [.reserved, .reserved], [none, none], [none, none], [], []⟩,
⟨3, [.reserved, .reserved], [none, none], [none, none], [(0, 0)], []⟩,
⟨4, [.reserved, .reserved], [none, none], [none, none], [(0, 0), (1, 1)], []⟩,
⟨3, [.reserved, .reserved], [(some 0), none], [none, none], [], []⟩,
⟨4, [.reserved, .reserved], [(some 0), none], [none, none], [(1, 1)], []⟩,
⟨4, [.reserved, .reserved], [none, (some 1)], [none, none], [(0, 0)], []⟩,
⟨4, [.reserved, .reserved], [(some 0), (some 1)], [none, none], [], []⟩,
⟨5, [.reserved, .reserved], [(some 0), (some 1)], [none, none], [], []⟩,
⟨6, [.published, .reserved], [(some 0), (some 1)], [none, none], [], []⟩,
⟨7, [.published, .published], [(some 0), (some 1)], [none, none], [], []⟩,
⟨8, [.acquired, .published], [(some 0), (some 1)], [none, none], [], []⟩,
⟨9, [.acquired, .acquired], [(some 0), (some 1)], [none, none], [], []⟩,
⟨10, [.acquired, .acquired], [(some 0), (some 1)], [none, none], [], [(0, 0, 0)]⟩,
⟨11, [.acquired, .acquired], [(some 0), (some 1)], [none, none], [], [(0, 0, 0), (1, 1, 1)]⟩,
⟨10, [.acquired, .acquired], [(some 0), (some 1)], [(some 0), none], [], []⟩,
⟨11, [.acquired, .acquired], [(some 0), (some 1)], [(some 0), none], [], [(1, 1, 1)]⟩,
⟨11, [.acquired, .acquired], [(some 0), (some 1)], [none, (some 1)], [], [(0, 0, 0)]⟩,
⟨11, [.acquired, .acquired], [(some 0), (some 1)], [(some 0), (some 1)], [], []⟩,
⟨12, [.acquired, .acquired], [(some 0), (some 1)], [(some 0), (some 1)], [], []⟩,
⟨13, [.free, .acquired], [none, (some 1)], [(some 0), (some 1)], [], []⟩,
⟨14, [.free, .free], [none, none], [(some 0), (some 1)], [], []⟩]
theorem accepted : checkCertificate task program nodes = true := by decide +kernel
theorem safe (s : State) (h : Reachable task program s) :
    issueSafe task program s = true :=
  accepted_issueSafe task program nodes accepted s h
theorem progress (s : State) (h : Reachable task program s) :
    Acc (fun q s => q ∈ successors task program s) s :=
  accepted_accessible task program nodes accepted s h
theorem outputs {α : Type} (input : Nat → α) (s : State)
    (h : Reachable task program s) (done : successors task program s = []) :
    outputPayloads input s = task.expected.map (fun i => some (input i)) :=
  accepted_output_all_payloads task program nodes accepted s h done input
#print axioms accepted
#print axioms safe
#print axioms progress
#print axioms outputs
end SubmittedProtocol
