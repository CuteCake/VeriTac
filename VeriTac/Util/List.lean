/-
  VeriTac.Util.List
  Utility lemmas for List operations.
-/

namespace VeriTac.Util

/-- Swap two elements in a list if both indices are in range. -/
def List.swap {α : Type} (l : List α) (i j : Nat) : List α :=
  if hi : i < l.length then
    if hj : j < l.length then
      let vi := l[i]
      let vj := l[j]
      ((l.set i vj).set j vi)
    else l
  else l

end VeriTac.Util
