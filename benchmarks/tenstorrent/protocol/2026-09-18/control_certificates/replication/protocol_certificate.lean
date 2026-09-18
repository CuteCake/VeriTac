import VeriTac.Tenstorrent.Protocol
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace SubmittedProtocol
open VeriTac.Tenstorrent
def task : Task := ⟨1, [0, 0, 0, 0], 1, 512⟩
def program : List Instr := [.reserve 0, .read 0 0, .waitReads, .publish 0, .acquire 0, .write 0 0, .waitWrites, .release 0, .reserve 0, .read 0 0, .waitReads, .publish 0, .acquire 0, .write 0 1, .waitWrites, .release 0, .reserve 0, .read 0 0, .waitReads, .publish 0, .acquire 0, .write 0 2, .waitWrites, .release 0, .reserve 0, .read 0 0, .waitReads, .publish 0, .acquire 0, .write 0 3, .waitWrites, .release 0]
def nodes : List State := [
⟨0, [.free], [none], [none, none, none, none], [], []⟩,
⟨1, [.reserved], [none], [none, none, none, none], [], []⟩,
⟨2, [.reserved], [none], [none, none, none, none], [(0, 0)], []⟩,
⟨2, [.reserved], [(some 0)], [none, none, none, none], [], []⟩,
⟨3, [.reserved], [(some 0)], [none, none, none, none], [], []⟩,
⟨4, [.published], [(some 0)], [none, none, none, none], [], []⟩,
⟨5, [.acquired], [(some 0)], [none, none, none, none], [], []⟩,
⟨6, [.acquired], [(some 0)], [none, none, none, none], [], [(0, 0, 0)]⟩,
⟨6, [.acquired], [(some 0)], [(some 0), none, none, none], [], []⟩,
⟨7, [.acquired], [(some 0)], [(some 0), none, none, none], [], []⟩,
⟨8, [.free], [none], [(some 0), none, none, none], [], []⟩,
⟨9, [.reserved], [none], [(some 0), none, none, none], [], []⟩,
⟨10, [.reserved], [none], [(some 0), none, none, none], [(0, 0)], []⟩,
⟨10, [.reserved], [(some 0)], [(some 0), none, none, none], [], []⟩,
⟨11, [.reserved], [(some 0)], [(some 0), none, none, none], [], []⟩,
⟨12, [.published], [(some 0)], [(some 0), none, none, none], [], []⟩,
⟨13, [.acquired], [(some 0)], [(some 0), none, none, none], [], []⟩,
⟨14, [.acquired], [(some 0)], [(some 0), none, none, none], [], [(0, 1, 0)]⟩,
⟨14, [.acquired], [(some 0)], [(some 0), (some 0), none, none], [], []⟩,
⟨15, [.acquired], [(some 0)], [(some 0), (some 0), none, none], [], []⟩,
⟨16, [.free], [none], [(some 0), (some 0), none, none], [], []⟩,
⟨17, [.reserved], [none], [(some 0), (some 0), none, none], [], []⟩,
⟨18, [.reserved], [none], [(some 0), (some 0), none, none], [(0, 0)], []⟩,
⟨18, [.reserved], [(some 0)], [(some 0), (some 0), none, none], [], []⟩,
⟨19, [.reserved], [(some 0)], [(some 0), (some 0), none, none], [], []⟩,
⟨20, [.published], [(some 0)], [(some 0), (some 0), none, none], [], []⟩,
⟨21, [.acquired], [(some 0)], [(some 0), (some 0), none, none], [], []⟩,
⟨22, [.acquired], [(some 0)], [(some 0), (some 0), none, none], [], [(0, 2, 0)]⟩,
⟨22, [.acquired], [(some 0)], [(some 0), (some 0), (some 0), none], [], []⟩,
⟨23, [.acquired], [(some 0)], [(some 0), (some 0), (some 0), none], [], []⟩,
⟨24, [.free], [none], [(some 0), (some 0), (some 0), none], [], []⟩,
⟨25, [.reserved], [none], [(some 0), (some 0), (some 0), none], [], []⟩,
⟨26, [.reserved], [none], [(some 0), (some 0), (some 0), none], [(0, 0)], []⟩,
⟨26, [.reserved], [(some 0)], [(some 0), (some 0), (some 0), none], [], []⟩,
⟨27, [.reserved], [(some 0)], [(some 0), (some 0), (some 0), none], [], []⟩,
⟨28, [.published], [(some 0)], [(some 0), (some 0), (some 0), none], [], []⟩,
⟨29, [.acquired], [(some 0)], [(some 0), (some 0), (some 0), none], [], []⟩,
⟨30, [.acquired], [(some 0)], [(some 0), (some 0), (some 0), none], [], [(0, 3, 0)]⟩,
⟨30, [.acquired], [(some 0)], [(some 0), (some 0), (some 0), (some 0)], [], []⟩,
⟨31, [.acquired], [(some 0)], [(some 0), (some 0), (some 0), (some 0)], [], []⟩,
⟨32, [.free], [none], [(some 0), (some 0), (some 0), (some 0)], [], []⟩]
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
