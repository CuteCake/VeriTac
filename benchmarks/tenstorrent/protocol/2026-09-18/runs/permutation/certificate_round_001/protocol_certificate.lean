import VeriTac.Tenstorrent.Protocol
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace SubmittedProtocol
open VeriTac.Tenstorrent
def task : Task := ⟨4, [2, 0, 3, 1], 2, 512⟩
def program : List Instr := [.reserve 0, .read 0 2, .reserve 1, .read 1 0, .waitReads, .publish 0, .acquire 0, .write 0 0, .publish 1, .acquire 1, .write 1 1, .waitWrites, .release 0, .release 1, .reserve 0, .read 0 3, .reserve 1, .read 1 1, .waitReads, .publish 0, .acquire 0, .write 0 2, .publish 1, .acquire 1, .write 1 3, .waitWrites, .release 0, .release 1]
def nodes : List State := [
⟨0, [.free, .free], [none, none], [none, none, none, none], [], []⟩,
⟨1, [.reserved, .free], [none, none], [none, none, none, none], [], []⟩,
⟨2, [.reserved, .free], [none, none], [none, none, none, none], [(0, 2)], []⟩,
⟨3, [.reserved, .reserved], [none, none], [none, none, none, none], [(0, 2)], []⟩,
⟨2, [.reserved, .free], [(some 2), none], [none, none, none, none], [], []⟩,
⟨4, [.reserved, .reserved], [none, none], [none, none, none, none], [(0, 2), (1, 0)], []⟩,
⟨3, [.reserved, .reserved], [(some 2), none], [none, none, none, none], [], []⟩,
⟨4, [.reserved, .reserved], [(some 2), none], [none, none, none, none], [(1, 0)], []⟩,
⟨4, [.reserved, .reserved], [none, (some 0)], [none, none, none, none], [(0, 2)], []⟩,
⟨4, [.reserved, .reserved], [(some 2), (some 0)], [none, none, none, none], [], []⟩,
⟨5, [.reserved, .reserved], [(some 2), (some 0)], [none, none, none, none], [], []⟩,
⟨6, [.published, .reserved], [(some 2), (some 0)], [none, none, none, none], [], []⟩,
⟨7, [.acquired, .reserved], [(some 2), (some 0)], [none, none, none, none], [], []⟩,
⟨8, [.acquired, .reserved], [(some 2), (some 0)], [none, none, none, none], [], [(0, 0, 2)]⟩,
⟨9, [.acquired, .published], [(some 2), (some 0)], [none, none, none, none], [], [(0, 0, 2)]⟩,
⟨8, [.acquired, .reserved], [(some 2), (some 0)], [(some 2), none, none, none], [], []⟩,
⟨10, [.acquired, .acquired], [(some 2), (some 0)], [none, none, none, none], [], [(0, 0, 2)]⟩,
⟨9, [.acquired, .published], [(some 2), (some 0)], [(some 2), none, none, none], [], []⟩,
⟨11, [.acquired, .acquired], [(some 2), (some 0)], [none, none, none, none], [], [(0, 0, 2), (1, 1, 0)]⟩,
⟨10, [.acquired, .acquired], [(some 2), (some 0)], [(some 2), none, none, none], [], []⟩,
⟨11, [.acquired, .acquired], [(some 2), (some 0)], [(some 2), none, none, none], [], [(1, 1, 0)]⟩,
⟨11, [.acquired, .acquired], [(some 2), (some 0)], [none, (some 0), none, none], [], [(0, 0, 2)]⟩,
⟨11, [.acquired, .acquired], [(some 2), (some 0)], [(some 2), (some 0), none, none], [], []⟩,
⟨12, [.acquired, .acquired], [(some 2), (some 0)], [(some 2), (some 0), none, none], [], []⟩,
⟨13, [.free, .acquired], [none, (some 0)], [(some 2), (some 0), none, none], [], []⟩,
⟨14, [.free, .free], [none, none], [(some 2), (some 0), none, none], [], []⟩,
⟨15, [.reserved, .free], [none, none], [(some 2), (some 0), none, none], [], []⟩,
⟨16, [.reserved, .free], [none, none], [(some 2), (some 0), none, none], [(0, 3)], []⟩,
⟨17, [.reserved, .reserved], [none, none], [(some 2), (some 0), none, none], [(0, 3)], []⟩,
⟨16, [.reserved, .free], [(some 3), none], [(some 2), (some 0), none, none], [], []⟩,
⟨18, [.reserved, .reserved], [none, none], [(some 2), (some 0), none, none], [(0, 3), (1, 1)], []⟩,
⟨17, [.reserved, .reserved], [(some 3), none], [(some 2), (some 0), none, none], [], []⟩,
⟨18, [.reserved, .reserved], [(some 3), none], [(some 2), (some 0), none, none], [(1, 1)], []⟩,
⟨18, [.reserved, .reserved], [none, (some 1)], [(some 2), (some 0), none, none], [(0, 3)], []⟩,
⟨18, [.reserved, .reserved], [(some 3), (some 1)], [(some 2), (some 0), none, none], [], []⟩,
⟨19, [.reserved, .reserved], [(some 3), (some 1)], [(some 2), (some 0), none, none], [], []⟩,
⟨20, [.published, .reserved], [(some 3), (some 1)], [(some 2), (some 0), none, none], [], []⟩,
⟨21, [.acquired, .reserved], [(some 3), (some 1)], [(some 2), (some 0), none, none], [], []⟩,
⟨22, [.acquired, .reserved], [(some 3), (some 1)], [(some 2), (some 0), none, none], [], [(0, 2, 3)]⟩,
⟨23, [.acquired, .published], [(some 3), (some 1)], [(some 2), (some 0), none, none], [], [(0, 2, 3)]⟩,
⟨22, [.acquired, .reserved], [(some 3), (some 1)], [(some 2), (some 0), (some 3), none], [], []⟩,
⟨24, [.acquired, .acquired], [(some 3), (some 1)], [(some 2), (some 0), none, none], [], [(0, 2, 3)]⟩,
⟨23, [.acquired, .published], [(some 3), (some 1)], [(some 2), (some 0), (some 3), none], [], []⟩,
⟨25, [.acquired, .acquired], [(some 3), (some 1)], [(some 2), (some 0), none, none], [], [(0, 2, 3), (1, 3, 1)]⟩,
⟨24, [.acquired, .acquired], [(some 3), (some 1)], [(some 2), (some 0), (some 3), none], [], []⟩,
⟨25, [.acquired, .acquired], [(some 3), (some 1)], [(some 2), (some 0), (some 3), none], [], [(1, 3, 1)]⟩,
⟨25, [.acquired, .acquired], [(some 3), (some 1)], [(some 2), (some 0), none, (some 1)], [], [(0, 2, 3)]⟩,
⟨25, [.acquired, .acquired], [(some 3), (some 1)], [(some 2), (some 0), (some 3), (some 1)], [], []⟩,
⟨26, [.acquired, .acquired], [(some 3), (some 1)], [(some 2), (some 0), (some 3), (some 1)], [], []⟩,
⟨27, [.free, .acquired], [none, (some 1)], [(some 2), (some 0), (some 3), (some 1)], [], []⟩,
⟨28, [.free, .free], [none, none], [(some 2), (some 0), (some 3), (some 1)], [], []⟩]
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
