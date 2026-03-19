/-
  VeriTac.Util.Finset
  Utility lemmas for sum partitioning used by tiling proofs.
-/
import VeriTac.IR.Denote

namespace VeriTac.Util

/-- Partition a fold over Fin N into blocks of size `bs`.
    The fold over the full range equals nested folds over blocks. -/
theorem foldFin_partition_blocks {α : Type} (f : α → α → α) (init : α)
    (N bs : Nat) (_hbs : bs > 0) (g : Fin N → α) :
    foldFin f init N g =
    foldFin f init ((N + bs - 1) / bs) (fun (t : Fin ((N + bs - 1) / bs)) =>
      foldFin f init (min bs (N - t.val * bs)) (fun (k : Fin (min bs (N - t.val * bs))) =>
        g ⟨t.val * bs + k.val, by sorry⟩)) := by
  sorry

end VeriTac.Util
