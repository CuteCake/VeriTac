/-
  VeriTac.AttentionScheduling
  Smoke test for the creative scheduling / scratch-lifetime structural proofs.
  Imports `VeriTac.Attention.Scheduling` (a full typecheck) and exercises the
  theorems on concrete instances: `Fin.rev` involution / bijection (exactly-once
  coverage), value preservation and explicit destination (`reverseWrite`), the
  time-disjoint MM1-then-epilogue shared-capacity example, and the drained /
  non-drained pending-copy models.

  No `axiom`, `sorry`, or `native_decide` here; all statements are proved from
  the module's theorems.
-/
import VeriTac.Attention.Scheduling

namespace VeriTac.Attention.Scheduling

open Function

/- Reverse scheduling is an involution on a concrete slot. -/
example : reverseSlot 8 (reverseSlot 8 (3 : Fin 8)) = (3 : Fin 8) :=
  reverseSlot_involution 8 (3 : Fin 8)

/- Reverse scheduling is a bijection (exactly-once coverage of all slots). -/
example : Bijective (reverseSlot 8) :=
  reverseSlot_bijective 8

/- Two different logical blocks never share a physical slot. -/
example : reverseSlot 8 (3 : Fin 8) ≠ reverseSlot 8 (5 : Fin 8) :=
  reverse_distinct_blocks_distinct_slots 8 (by decide)

/- Value preservation at the explicit destination slot: block 3's output
   appears at slot `reverseSlot 8 3`, unchanged. -/
example :
    scheduledOutput Fin.rev (fun b : Fin 8 => b.val) (reverseSlot 8 (3 : Fin 8))
      = (3 : Fin 8).val :=
  reverse_value_preserved (fun b : Fin 8 => b.val) (3 : Fin 8)

/- Row ownership is preserved: reversing a block yields the same owned rows. -/
example : ownedRows 32 8 (3 : Fin 8) = ownedRows 32 8 (reverseSlot 8 (reverseSlot 8 (3 : Fin 8))) :=
  reverse_keeps_rows 32 8 (3 : Fin 8)

/- Explicit output address: the unique slot `Fin.rev b` writes destination `b`
   with value `blockOutput b`. -/
example :
    reverseWrite (fun b : Fin 8 => b.val) (Fin.rev (3 : Fin 8)) = ((3 : Fin 8), (3 : Fin 8).val) :=
  reverseWrite_destination (fun b : Fin 8 => b.val) (3 : Fin 8)

/- Every destination is written by exactly one physical slot. -/
example (b : Fin 8) :
    ∃! p : Fin 8, (reverseWrite (fun b : Fin 8 => b.val) p).1 = b :=
  reverseWrite_dest_unique (fun b : Fin 8 => b.val) b

/- Scratch lifetimes: MM1 then epilogue are time-disjoint (no time is live in
   both), for any pending-copy functions. -/
example (t : Nat) :
    ¬ (LiveAt (mm1Phase 4096 256 pendingfun0) t ∧
       LiveAt (epiloguePhase 2048 256 512 pendingfun0) t) :=
  mm1_epilogue_no_overlap 4096 2048 256 512 pendingfun0 pendingfun0 t

/- A shared buffer of capacity `max 4096 2048` serves both phases. -/
example :
    (mm1Phase 4096 256 pendingfun0).size ≤ max 4096 2048 ∧
      (epiloguePhase 2048 256 512 pendingfun0).size ≤ max 4096 2048 :=
  mm1_epilogue_shared_capacity 4096 2048 256 512 pendingfun0 pendingfun0

/- The shared capacity never exceeds the sum of the two separate allocations. -/
example : max 4096 2048 ≤ 4096 + 2048 :=
  max_capacity_le_sum (mm1Phase 4096 256 pendingfun0) (epiloguePhase 2048 256 512 pendingfun0)

/- Positive pending model: pendingfun0 is drained, so a later phase sees zero
   outstanding copies from MM1. -/
example (sizeMM1 sizeEpilogue m1_end e_end t : Nat)
    (ht : LiveAt (epiloguePhase sizeEpilogue m1_end e_end pendingfun0) t) :
    (mm1Phase sizeMM1 m1_end pendingfun0).pending t = 0 := by
  exact drained_implies_pending_zero_in_later_phase
    (mm1Phase sizeMM1 m1_end pendingfun0)
    (epiloguePhase sizeEpilogue m1_end e_end pendingfun0)
    (le_rfl) (pendingfun0_drained sizeMM1 m1_end) t ht

/- Negative pending model: pendingfun1 is not drained (the shared-buffer
   guarantee cannot fire). -/
example (size m1_end : Nat) :
    ¬ AsyncDrained { lo := 0, hi := m1_end, size := size, pending := pendingfun1 } :=
  pendingfun1_not_drained size m1_end

end VeriTac.Attention.Scheduling
