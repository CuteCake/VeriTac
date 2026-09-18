import VeriTac.Tenstorrent.Protocol
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace SubmittedProtocol
open VeriTac.Tenstorrent
def task : Task := ⟨4, [2, 0, 3, 1], 2, 512⟩
def program : List Instr := [.reserve 0, .read 0 2, .waitReads, .publish 0, .acquire 0, .write 0 0, .waitWrites, .release 0, .reserve 0, .read 0 0, .waitReads, .publish 0, .acquire 0, .write 0 1, .waitWrites, .release 0, .reserve 0, .read 0 3, .waitReads, .publish 0, .acquire 0, .write 0 2, .waitWrites, .release 0, .reserve 0, .read 0 1, .waitReads, .publish 0, .acquire 0, .write 0 3, .waitWrites, .release 0]
def nodes : List State := [
⟨0, [.free, .free], [none, none], [none, none, none, none], [], []⟩,
⟨1, [.reserved, .free], [none, none], [none, none, none, none], [], []⟩,
⟨2, [.reserved, .free], [none, none], [none, none, none, none], [(0, 2)], []⟩,
⟨2, [.reserved, .free], [(some 2), none], [none, none, none, none], [], []⟩,
⟨3, [.reserved, .free], [(some 2), none], [none, none, none, none], [], []⟩,
⟨4, [.published, .free], [(some 2), none], [none, none, none, none], [], []⟩,
⟨5, [.acquired, .free], [(some 2), none], [none, none, none, none], [], []⟩,
⟨6, [.acquired, .free], [(some 2), none], [none, none, none, none], [], [(0, 0, 2)]⟩,
⟨6, [.acquired, .free], [(some 2), none], [(some 2), none, none, none], [], []⟩,
⟨7, [.acquired, .free], [(some 2), none], [(some 2), none, none, none], [], []⟩,
⟨8, [.free, .free], [none, none], [(some 2), none, none, none], [], []⟩,
⟨9, [.reserved, .free], [none, none], [(some 2), none, none, none], [], []⟩,
⟨10, [.reserved, .free], [none, none], [(some 2), none, none, none], [(0, 0)], []⟩,
⟨10, [.reserved, .free], [(some 0), none], [(some 2), none, none, none], [], []⟩,
⟨11, [.reserved, .free], [(some 0), none], [(some 2), none, none, none], [], []⟩,
⟨12, [.published, .free], [(some 0), none], [(some 2), none, none, none], [], []⟩,
⟨13, [.acquired, .free], [(some 0), none], [(some 2), none, none, none], [], []⟩,
⟨14, [.acquired, .free], [(some 0), none], [(some 2), none, none, none], [], [(0, 1, 0)]⟩,
⟨14, [.acquired, .free], [(some 0), none], [(some 2), (some 0), none, none], [], []⟩,
⟨15, [.acquired, .free], [(some 0), none], [(some 2), (some 0), none, none], [], []⟩,
⟨16, [.free, .free], [none, none], [(some 2), (some 0), none, none], [], []⟩,
⟨17, [.reserved, .free], [none, none], [(some 2), (some 0), none, none], [], []⟩,
⟨18, [.reserved, .free], [none, none], [(some 2), (some 0), none, none], [(0, 3)], []⟩,
⟨18, [.reserved, .free], [(some 3), none], [(some 2), (some 0), none, none], [], []⟩,
⟨19, [.reserved, .free], [(some 3), none], [(some 2), (some 0), none, none], [], []⟩,
⟨20, [.published, .free], [(some 3), none], [(some 2), (some 0), none, none], [], []⟩,
⟨21, [.acquired, .free], [(some 3), none], [(some 2), (some 0), none, none], [], []⟩,
⟨22, [.acquired, .free], [(some 3), none], [(some 2), (some 0), none, none], [], [(0, 2, 3)]⟩,
⟨22, [.acquired, .free], [(some 3), none], [(some 2), (some 0), (some 3), none], [], []⟩,
⟨23, [.acquired, .free], [(some 3), none], [(some 2), (some 0), (some 3), none], [], []⟩,
⟨24, [.free, .free], [none, none], [(some 2), (some 0), (some 3), none], [], []⟩,
⟨25, [.reserved, .free], [none, none], [(some 2), (some 0), (some 3), none], [], []⟩,
⟨26, [.reserved, .free], [none, none], [(some 2), (some 0), (some 3), none], [(0, 1)], []⟩,
⟨26, [.reserved, .free], [(some 1), none], [(some 2), (some 0), (some 3), none], [], []⟩,
⟨27, [.reserved, .free], [(some 1), none], [(some 2), (some 0), (some 3), none], [], []⟩,
⟨28, [.published, .free], [(some 1), none], [(some 2), (some 0), (some 3), none], [], []⟩,
⟨29, [.acquired, .free], [(some 1), none], [(some 2), (some 0), (some 3), none], [], []⟩,
⟨30, [.acquired, .free], [(some 1), none], [(some 2), (some 0), (some 3), none], [], [(0, 3, 1)]⟩,
⟨30, [.acquired, .free], [(some 1), none], [(some 2), (some 0), (some 3), (some 1)], [], []⟩,
⟨31, [.acquired, .free], [(some 1), none], [(some 2), (some 0), (some 3), (some 1)], [], []⟩,
⟨32, [.free, .free], [none, none], [(some 2), (some 0), (some 3), (some 1)], [], []⟩]
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
