import VeriTac.Tenstorrent.Protocol
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace SubmittedProtocol
open VeriTac.Tenstorrent
def task : Task := ⟨2, [1, 0, 1, 0], 2, 512⟩
def program : List Instr := [.reserve 0, .read 0 1, .reserve 1, .read 1 0, .waitReads, .publish 0, .acquire 0, .write 0 0, .write 0 2, .publish 1, .acquire 1, .write 1 1, .write 1 3, .waitWrites, .release 0, .release 1]
def nodes : List State := [
⟨0, [.free, .free], [none, none], [none, none, none, none], [], []⟩,
⟨1, [.reserved, .free], [none, none], [none, none, none, none], [], []⟩,
⟨2, [.reserved, .free], [none, none], [none, none, none, none], [(0, 1)], []⟩,
⟨3, [.reserved, .reserved], [none, none], [none, none, none, none], [(0, 1)], []⟩,
⟨2, [.reserved, .free], [(some 1), none], [none, none, none, none], [], []⟩,
⟨4, [.reserved, .reserved], [none, none], [none, none, none, none], [(0, 1), (1, 0)], []⟩,
⟨3, [.reserved, .reserved], [(some 1), none], [none, none, none, none], [], []⟩,
⟨4, [.reserved, .reserved], [(some 1), none], [none, none, none, none], [(1, 0)], []⟩,
⟨4, [.reserved, .reserved], [none, (some 0)], [none, none, none, none], [(0, 1)], []⟩,
⟨4, [.reserved, .reserved], [(some 1), (some 0)], [none, none, none, none], [], []⟩,
⟨5, [.reserved, .reserved], [(some 1), (some 0)], [none, none, none, none], [], []⟩,
⟨6, [.published, .reserved], [(some 1), (some 0)], [none, none, none, none], [], []⟩,
⟨7, [.acquired, .reserved], [(some 1), (some 0)], [none, none, none, none], [], []⟩,
⟨8, [.acquired, .reserved], [(some 1), (some 0)], [none, none, none, none], [], [(0, 0, 1)]⟩,
⟨9, [.acquired, .reserved], [(some 1), (some 0)], [none, none, none, none], [], [(0, 0, 1), (0, 2, 1)]⟩,
⟨8, [.acquired, .reserved], [(some 1), (some 0)], [(some 1), none, none, none], [], []⟩,
⟨10, [.acquired, .published], [(some 1), (some 0)], [none, none, none, none], [], [(0, 0, 1), (0, 2, 1)]⟩,
⟨9, [.acquired, .reserved], [(some 1), (some 0)], [(some 1), none, none, none], [], [(0, 2, 1)]⟩,
⟨9, [.acquired, .reserved], [(some 1), (some 0)], [none, none, (some 1), none], [], [(0, 0, 1)]⟩,
⟨11, [.acquired, .acquired], [(some 1), (some 0)], [none, none, none, none], [], [(0, 0, 1), (0, 2, 1)]⟩,
⟨10, [.acquired, .published], [(some 1), (some 0)], [(some 1), none, none, none], [], [(0, 2, 1)]⟩,
⟨10, [.acquired, .published], [(some 1), (some 0)], [none, none, (some 1), none], [], [(0, 0, 1)]⟩,
⟨9, [.acquired, .reserved], [(some 1), (some 0)], [(some 1), none, (some 1), none], [], []⟩,
⟨12, [.acquired, .acquired], [(some 1), (some 0)], [none, none, none, none], [], [(0, 0, 1), (0, 2, 1), (1, 1, 0)]⟩,
⟨11, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), none, none, none], [], [(0, 2, 1)]⟩,
⟨11, [.acquired, .acquired], [(some 1), (some 0)], [none, none, (some 1), none], [], [(0, 0, 1)]⟩,
⟨10, [.acquired, .published], [(some 1), (some 0)], [(some 1), none, (some 1), none], [], []⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [none, none, none, none], [], [(0, 0, 1), (0, 2, 1), (1, 1, 0), (1, 3, 0)]⟩,
⟨12, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), none, none, none], [], [(0, 2, 1), (1, 1, 0)]⟩,
⟨12, [.acquired, .acquired], [(some 1), (some 0)], [none, none, (some 1), none], [], [(0, 0, 1), (1, 1, 0)]⟩,
⟨12, [.acquired, .acquired], [(some 1), (some 0)], [none, (some 0), none, none], [], [(0, 0, 1), (0, 2, 1)]⟩,
⟨11, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), none, (some 1), none], [], []⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), none, none, none], [], [(0, 2, 1), (1, 1, 0), (1, 3, 0)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [none, none, (some 1), none], [], [(0, 0, 1), (1, 1, 0), (1, 3, 0)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [none, (some 0), none, none], [], [(0, 0, 1), (0, 2, 1), (1, 3, 0)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [none, none, none, (some 0)], [], [(0, 0, 1), (0, 2, 1), (1, 1, 0)]⟩,
⟨12, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), none, (some 1), none], [], [(1, 1, 0)]⟩,
⟨12, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), (some 0), none, none], [], [(0, 2, 1)]⟩,
⟨12, [.acquired, .acquired], [(some 1), (some 0)], [none, (some 0), (some 1), none], [], [(0, 0, 1)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), none, (some 1), none], [], [(1, 1, 0), (1, 3, 0)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), (some 0), none, none], [], [(0, 2, 1), (1, 3, 0)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), none, none, (some 0)], [], [(0, 2, 1), (1, 1, 0)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [none, (some 0), (some 1), none], [], [(0, 0, 1), (1, 3, 0)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [none, none, (some 1), (some 0)], [], [(0, 0, 1), (1, 1, 0)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [none, (some 0), none, (some 0)], [], [(0, 0, 1), (0, 2, 1)]⟩,
⟨12, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), (some 0), (some 1), none], [], []⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), (some 0), (some 1), none], [], [(1, 3, 0)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), none, (some 1), (some 0)], [], [(1, 1, 0)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), (some 0), none, (some 0)], [], [(0, 2, 1)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [none, (some 0), (some 1), (some 0)], [], [(0, 0, 1)]⟩,
⟨13, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), (some 0), (some 1), (some 0)], [], []⟩,
⟨14, [.acquired, .acquired], [(some 1), (some 0)], [(some 1), (some 0), (some 1), (some 0)], [], []⟩,
⟨15, [.free, .acquired], [none, (some 0)], [(some 1), (some 0), (some 1), (some 0)], [], []⟩,
⟨16, [.free, .free], [none, none], [(some 1), (some 0), (some 1), (some 0)], [], []⟩]
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
