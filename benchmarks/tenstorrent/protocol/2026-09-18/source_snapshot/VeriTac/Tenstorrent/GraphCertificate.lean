import Mathlib

namespace VeriTac.ProtocolGraph

variable {S : Type} [DecidableEq S]

/-- A finite certificate proposes an overapproximation of reachable states.
The executable checker verifies closure; the producer of the list is untrusted. -/
def check (initial : S) (next : S → List S) (good : S → Bool) (nodes : List S) : Bool :=
  decide (initial ∈ nodes) && nodes.all (fun s => good s && (next s).all (fun q => decide (q ∈ nodes)))

inductive Reach (initial : S) (next : S → List S) : S → Prop where
  | initial : Reach initial next initial
  | step {s q : S} : Reach initial next s → q ∈ next s → Reach initial next q

theorem check_sound (initial : S) (next : S → List S) (good : S → Bool) (nodes : List S)
    (h : check initial next good nodes = true) :
    ∀ s, Reach initial next s → good s = true := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  have mem : ∀ s, Reach initial next s → s ∈ nodes := by
    intro s hs
    induction hs with
    | initial => exact h.1
    | @step s q hs hedge ih => exact (h.2 s ih).2 q hedge
  intro s hs
  exact (h.2 s (mem s hs)).1

#print axioms check_sound
end VeriTac.ProtocolGraph
