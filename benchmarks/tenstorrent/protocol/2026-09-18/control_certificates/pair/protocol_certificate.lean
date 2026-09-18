import VeriTac.Tenstorrent.Protocol
set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace SubmittedProtocol
open VeriTac.Tenstorrent
def task : Task := ⟨2, [0, 1], 2, 512⟩
def program : List Instr := [.reserve 0, .read 0 0, .waitReads, .publish 0, .acquire 0, .write 0 0, .waitWrites, .release 0, .reserve 0, .read 0 1, .waitReads, .publish 0, .acquire 0, .write 0 1, .waitWrites, .release 0]
def nodes : List State := [
⟨0, [.free, .free], [none, none], [none, none], [], []⟩,
⟨1, [.reserved, .free], [none, none], [none, none], [], []⟩,
⟨2, [.reserved, .free], [none, none], [none, none], [(0, 0)], []⟩,
⟨2, [.reserved, .free], [(some 0), none], [none, none], [], []⟩,
⟨3, [.reserved, .free], [(some 0), none], [none, none], [], []⟩,
⟨4, [.published, .free], [(some 0), none], [none, none], [], []⟩,
⟨5, [.acquired, .free], [(some 0), none], [none, none], [], []⟩,
⟨6, [.acquired, .free], [(some 0), none], [none, none], [], [(0, 0, 0)]⟩,
⟨6, [.acquired, .free], [(some 0), none], [(some 0), none], [], []⟩,
⟨7, [.acquired, .free], [(some 0), none], [(some 0), none], [], []⟩,
⟨8, [.free, .free], [none, none], [(some 0), none], [], []⟩,
⟨9, [.reserved, .free], [none, none], [(some 0), none], [], []⟩,
⟨10, [.reserved, .free], [none, none], [(some 0), none], [(0, 1)], []⟩,
⟨10, [.reserved, .free], [(some 1), none], [(some 0), none], [], []⟩,
⟨11, [.reserved, .free], [(some 1), none], [(some 0), none], [], []⟩,
⟨12, [.published, .free], [(some 1), none], [(some 0), none], [], []⟩,
⟨13, [.acquired, .free], [(some 1), none], [(some 0), none], [], []⟩,
⟨14, [.acquired, .free], [(some 1), none], [(some 0), none], [], [(0, 1, 1)]⟩,
⟨14, [.acquired, .free], [(some 1), none], [(some 0), (some 1)], [], []⟩,
⟨15, [.acquired, .free], [(some 1), none], [(some 0), (some 1)], [], []⟩,
⟨16, [.free, .free], [none, none], [(some 0), (some 1)], [], []⟩]
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
