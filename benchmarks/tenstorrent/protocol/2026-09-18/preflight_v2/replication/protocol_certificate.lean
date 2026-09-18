import VeriTac.Tenstorrent.Protocol
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace SubmittedProtocol
open VeriTac.Tenstorrent
def task : Task := ⟨1, [0, 0, 0, 0], 1, 512⟩
def program : List Instr := [.reserve 0, .read 0 0, .waitReads, .publish 0, .acquire 0, .write 0 0, .write 0 1, .write 0 2, .write 0 3, .waitWrites, .release 0]
def nodes : List State := [
⟨0, [.free], [none], [none, none, none, none], [], []⟩,
⟨1, [.reserved], [none], [none, none, none, none], [], []⟩,
⟨2, [.reserved], [none], [none, none, none, none], [(0, 0)], []⟩,
⟨2, [.reserved], [(some 0)], [none, none, none, none], [], []⟩,
⟨3, [.reserved], [(some 0)], [none, none, none, none], [], []⟩,
⟨4, [.published], [(some 0)], [none, none, none, none], [], []⟩,
⟨5, [.acquired], [(some 0)], [none, none, none, none], [], []⟩,
⟨6, [.acquired], [(some 0)], [none, none, none, none], [], [(0, 0, 0)]⟩,
⟨7, [.acquired], [(some 0)], [none, none, none, none], [], [(0, 0, 0), (0, 1, 0)]⟩,
⟨6, [.acquired], [(some 0)], [(some 0), none, none, none], [], []⟩,
⟨8, [.acquired], [(some 0)], [none, none, none, none], [], [(0, 0, 0), (0, 1, 0), (0, 2, 0)]⟩,
⟨7, [.acquired], [(some 0)], [(some 0), none, none, none], [], [(0, 1, 0)]⟩,
⟨7, [.acquired], [(some 0)], [none, (some 0), none, none], [], [(0, 0, 0)]⟩,
⟨9, [.acquired], [(some 0)], [none, none, none, none], [], [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)]⟩,
⟨8, [.acquired], [(some 0)], [(some 0), none, none, none], [], [(0, 1, 0), (0, 2, 0)]⟩,
⟨8, [.acquired], [(some 0)], [none, (some 0), none, none], [], [(0, 0, 0), (0, 2, 0)]⟩,
⟨8, [.acquired], [(some 0)], [none, none, (some 0), none], [], [(0, 0, 0), (0, 1, 0)]⟩,
⟨7, [.acquired], [(some 0)], [(some 0), (some 0), none, none], [], []⟩,
⟨9, [.acquired], [(some 0)], [(some 0), none, none, none], [], [(0, 1, 0), (0, 2, 0), (0, 3, 0)]⟩,
⟨9, [.acquired], [(some 0)], [none, (some 0), none, none], [], [(0, 0, 0), (0, 2, 0), (0, 3, 0)]⟩,
⟨9, [.acquired], [(some 0)], [none, none, (some 0), none], [], [(0, 0, 0), (0, 1, 0), (0, 3, 0)]⟩,
⟨9, [.acquired], [(some 0)], [none, none, none, (some 0)], [], [(0, 0, 0), (0, 1, 0), (0, 2, 0)]⟩,
⟨8, [.acquired], [(some 0)], [(some 0), (some 0), none, none], [], [(0, 2, 0)]⟩,
⟨8, [.acquired], [(some 0)], [(some 0), none, (some 0), none], [], [(0, 1, 0)]⟩,
⟨8, [.acquired], [(some 0)], [none, (some 0), (some 0), none], [], [(0, 0, 0)]⟩,
⟨9, [.acquired], [(some 0)], [(some 0), (some 0), none, none], [], [(0, 2, 0), (0, 3, 0)]⟩,
⟨9, [.acquired], [(some 0)], [(some 0), none, (some 0), none], [], [(0, 1, 0), (0, 3, 0)]⟩,
⟨9, [.acquired], [(some 0)], [(some 0), none, none, (some 0)], [], [(0, 1, 0), (0, 2, 0)]⟩,
⟨9, [.acquired], [(some 0)], [none, (some 0), (some 0), none], [], [(0, 0, 0), (0, 3, 0)]⟩,
⟨9, [.acquired], [(some 0)], [none, (some 0), none, (some 0)], [], [(0, 0, 0), (0, 2, 0)]⟩,
⟨9, [.acquired], [(some 0)], [none, none, (some 0), (some 0)], [], [(0, 0, 0), (0, 1, 0)]⟩,
⟨8, [.acquired], [(some 0)], [(some 0), (some 0), (some 0), none], [], []⟩,
⟨9, [.acquired], [(some 0)], [(some 0), (some 0), (some 0), none], [], [(0, 3, 0)]⟩,
⟨9, [.acquired], [(some 0)], [(some 0), (some 0), none, (some 0)], [], [(0, 2, 0)]⟩,
⟨9, [.acquired], [(some 0)], [(some 0), none, (some 0), (some 0)], [], [(0, 1, 0)]⟩,
⟨9, [.acquired], [(some 0)], [none, (some 0), (some 0), (some 0)], [], [(0, 0, 0)]⟩,
⟨9, [.acquired], [(some 0)], [(some 0), (some 0), (some 0), (some 0)], [], []⟩,
⟨10, [.acquired], [(some 0)], [(some 0), (some 0), (some 0), (some 0)], [], []⟩,
⟨11, [.free], [none], [(some 0), (some 0), (some 0), (some 0)], [], []⟩]
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
