import VeriTac.Gemmini.Checked
import VeriTac.Gemmini.SymbolicSound
import VeriTac.Gemmini.Verification

namespace VeriTac.Gemmini
set_option maxRecDepth 8000
open Encoding

@[simp] private theorem decode_ex (bases : Bases) :
    commandOf bases ⟨0, 4575657221408489476, 281474976710656⟩ = some (.configEx 1) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_ld0 (bases : Bases) :
    commandOf bases ⟨0, 4575657221409472769, 16⟩ = some (.configLd 0 16 true) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_ld1 (bases : Bases) :
    commandOf bases ⟨0, 4575657221409472777, 16⟩ = some (.configLd 1 16 true) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_st (bases : Bases) :
    commandOf bases ⟨0, 2, 4575657221408424000⟩ = some (.configSt 64) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_a0 (bases : Bases) :
    commandOf bases ⟨2, bases.a, packedTile 0⟩ = some (.mvin 0 .a 0 0 16 16) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_a256 (bases : Bases) :
    commandOf bases ⟨2, bases.a + 256, packedTile 0⟩ = some (.mvin 0 .a 256 0 16 16) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_b0 (bases : Bases) :
    commandOf bases ⟨1, bases.b, packedTile 16⟩ = some (.mvin 1 .b 0 16 16 16) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_pre0 (bases : Bases) :
    commandOf bases ⟨6, packedTile 16, packedTile 2684354560⟩ = some (.preload 16 2684354560) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_pre16 (bases : Bases) :
    commandOf bases ⟨6, packedTile 4294967295, packedTile 2684354576⟩ = some (.preload 4294967295 2684354576) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_compute (bases : Bases) :
    commandOf bases ⟨4, packedTile 0, packedTile 4294967295⟩ = some (.compute false 0 4294967295) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_retained (bases : Bases) :
    commandOf bases ⟨5, packedTile 0, packedTile 4294967295⟩ = some (.compute true 0 4294967295) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_out0 (bases : Bases) :
    commandOf bases ⟨3, bases.c, packedTile 3758096384⟩ = some (.mvout 0 3758096384 16 16) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_out256 (bases : Bases) :
    commandOf bases ⟨3, bases.c + 1024, packedTile 3758096384⟩ = some (.mvout 256 3758096384 16 16) := by
  norm_num [commandOf, unpackTile, packedTile]

@[simp] private theorem decode_out16 (bases : Bases) :
    commandOf bases ⟨3, bases.c + 1024, packedTile 3758096400⟩ = some (.mvout 256 3758096400 16 16) := by
  norm_num [commandOf, unpackTile, packedTile]

set_option maxRecDepth 8000 in
set_option maxHeartbeats 1600000 in
theorem supported_packet_roundtrip (p : GemminiPlan) (program : Program) (bases : Bases)
    (h : checkSupportedProgram p program = true) :
    commandsOf bases (packetize bases program) = some program := by
  rcases Bool.or_eq_true_iff.mp h with h | h
  · simp only [checkProgram, Bool.and_eq_true, beq_iff_eq] at h
    have hs := h.2
    cases he : p.schedule <;> simp [he] at hs <;> subst program <;>
      simp [commandsOf, packetize, baseline16, reuse16, packetOf]
  · simp only [checkProgram32, Bool.and_eq_true, beq_iff_eq] at h
    have hs := h.2
    cases he : p.schedule <;> simp [he] at hs <;> subst program <;>
      simp [commandsOf, packetize, baseline32, reuse32, packetOf]

/-- Direct interpretation of the supplied bytes: RV64 execution -> decoded
Gemmini command operands -> Gemmini state transitions. No supplied program is
used to define executable behavior. The fence is reconstructed only after the
byte interpreter has established an actual terminal fence+return. -/
def runExecutable (p : GemminiPlan) (bases : Bases) (bytes : List UInt8)
    (a b : Nat → Int) : Option MachineState := do
  let packets ← (executeBytes bytes).toOption
  let commands ← commandsOf bases packets
  runProgram p a b commands

/-- The acceptance boundary connects the ACTUAL submitted executable bytes to
GEMM for all int8 inputs, under this declared target and loader contract. -/
theorem checkExecutable_sound (p : GemminiPlan) (program : Program) (bases : Bases)
    (bytes : List UInt8) (h : checkExecutable p program bases bytes = true)
    (a b : Nat → Int)
    (ha : ∀ i < p.m * p.k, GemminiExact.Int8 (a i))
    (hb : ∀ i < p.k * p.n, GemminiExact.Int8 (b i)) :
    checkBases bases (bufferSizes p) = true ∧
    ∃ s, runExecutable p bases bytes a b = some s ∧ s.drained = true ∧
      ∀ r < p.m, ∀ c < p.n,
        s.output (r * p.n + c) = gemm p a b r c ∧ s.written (r * p.n + c) = true := by
  obtain ⟨hcapProgram, hencoding⟩ := Bool.and_eq_true_iff.mp h
  have hprogram := (Bool.and_eq_true_iff.mp hcapProgram).2
  cases hd : Encoding.decode bases (bufferSizes p) bytes with
  | error message => simp [hd] at hencoding
  | ok packets =>
    have hdecoded : commandsOf bases packets = some program := by
      cases hc : commandsOf bases packets with
      | none => simp [hd, hc] at hencoding
      | some decoded =>
        have heq : decoded = program := of_decide_eq_true (by simpa [hd, hc] using hencoding)
        simpa [heq] using hc
    have hex : checkBases bases (bufferSizes p) = true ∧ executeBytes bytes = .ok packets := by
      unfold Encoding.decode at hd
      split at hd
      · exact ⟨by assumption, hd⟩
      · contradiction
    obtain ⟨s, hs, hdone, hout⟩ := Symbolic.check_sound p program hprogram a b ha hb
    refine ⟨hex.1, s, ?_, hdone, hout⟩
    simpa [runExecutable, hex.2, hdecoded, Except.toOption] using hs

end VeriTac.Gemmini
