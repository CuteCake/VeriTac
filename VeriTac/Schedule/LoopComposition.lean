/-
  VeriTac.Schedule.LoopComposition
  Composition lemmas for `execLoopIters`: congruence, splitting (append),
  commutation, permutation (reorder), partition into blocks (tile), and
  fusion. Every lemma is stated over an abstract iteration body
  `f : Env → Store → Option Store`, so each tactic proof only has to
  instantiate `f` with the execution of its loop body.
-/
import Mathlib.Tactic
import VeriTac.Schedule.LoopNest

namespace VeriTac

/-! ## Environment helpers -/

theorem Env.set_set_same (env : Env) (v : VarId) (a b : Int) :
    Env.set (Env.set env v a) v b = Env.set env v b := by
  funext z
  by_cases h : z = v
  · subst z; simp [Env.set]
  · simp [Env.set, h]

/-! ## Integer div/mod splitting -/

/-- For `0 ≤ j < M`, the exact division `q * M + j` splits as quotient `q`
    and remainder `j`. This is the arithmetic behind tile and fuse. -/
theorem int_mul_add_div_mod (q : Int) (M : Nat) (j : Int) (hj : 0 ≤ j)
    (hlt : j < (M : Int)) (hM : 0 < (M : Int)) :
    (q * (M : Int) + j) / (M : Int) = q ∧ (q * (M : Int) + j) % (M : Int) = j := by
  have hM0 : (M : Int) ≠ 0 := by omega
  have hj0 : j / (M : Int) = 0 := by
    rw [← Int.toNat_of_nonneg hj, ← Int.natCast_div, Nat.cast_eq_zero, Nat.div_eq_zero_iff]
    exact Or.inr (by omega)
  have hjm : j % (M : Int) = j := by
    rw [← Int.toNat_of_nonneg hj, ← Int.natCast_mod, Nat.mod_eq_of_lt (by omega),
      Int.toNat_of_nonneg hj]
  have hdiv : (q * (M : Int) + j) / (M : Int) = q := by
    rw [Int.add_comm (q * (M : Int)) j, Int.add_mul_ediv_right j q hM0, hj0]
    ring
  refine ⟨hdiv, ?_⟩
  have hk := Int.mul_ediv_add_emod (q * (M : Int) + j) (M : Int)
  rw [hdiv] at hk
  linarith

/-! ## Congruence lemmas -/

/-- Master congruence: if the two bodies agree on every iteration environment
    (which binds the loop variable), the folds agree. -/
theorem execLoopIters_congr (f1 f2 : Env → Store → Option Store) (E1 E2 : Env) (v : VarId)
    (h : ∀ (c : Int) (st : Store), f1 (Env.set E1 v c) st = f2 (Env.set E2 v c) st) :
    ∀ (n : Nat) (lo : Int) (st : Store),
      execLoopIters f1 E1 v lo n st = execLoopIters f2 E2 v lo n st := by
  intro n lo st
  induction n generalizing lo st with
  | zero => rfl
  | succ n ih =>
      simp only [execLoopIters]
      rw [h lo st]
      congr 1
      funext st'
      exact ih (lo + 1) st'

theorem execLoopIters_pure (env : Env) (v : VarId) :
    ∀ (n : Nat) (lo : Int) (st : Store),
      execLoopIters (fun _ s => some s) env v lo n st = some st := by
  intro n lo st
  induction n generalizing lo st with
  | zero => rfl
  | succ n ih =>
      simp only [execLoopIters]
      rw [funext (ih (lo + 1))]
      simp

/-- Pre-binding the loop's own variable in the base environment is a no-op:
    every iteration overwrites it. -/
theorem execLoopIters_self_bind (f : Env → Store → Option Store) (env : Env) (v : VarId)
    (x : Int) :
    ∀ (n : Nat) (lo : Int) (st : Store),
      execLoopIters f (Env.set env v x) v lo n st = execLoopIters f env v lo n st := by
  intro n lo st
  induction n generalizing lo st with
  | zero => rfl
  | succ n ih =>
      simp only [execLoopIters]
      rw [Env.set_set_same]
      congr 1
      funext st'
      exact ih (lo + 1) st'

/-! ## Splitting a fold into two consecutive folds -/

theorem execLoopIters_append (f : Env → Store → Option Store) (env : Env) (v : VarId)
    (a b : Nat) (lo : Int) (st : Store) :
    execLoopIters f env v lo (a + b) st =
    (execLoopIters f env v lo a st).bind (execLoopIters f env v (lo + (a : Int)) b) := by
  induction a generalizing lo st with
  | zero => simp [execLoopIters]
  | succ a ih =>
      have hab : (a + 1) + b = (a + b) + 1 := by omega
      have hlo : (lo : Int) + ((a + 1) : Nat) = lo + 1 + (a : Int) := by omega
      rw [hab, hlo]
      simp only [execLoopIters]
      rw [funext (ih (lo + 1))]
      simp only []
      exact (Option.bind_assoc _ _ _).symm

/-! ## Function-level append -/

/-! ## Kleisli composition of store transformers -/

/-- Kleisli composition of store transformers. Working with this combinator
    keeps every manipulation at the function level. -/
def kcomp (f g : Store → Option Store) : Store → Option Store := fun st => (f st).bind g

theorem kcomp_pure_left (f : Store → Option Store) :
    kcomp (fun st => some st) f = f := funext fun st => rfl

theorem kcomp_pure_right (f : Store → Option Store) :
    kcomp f (fun st => some st) = f := by
  funext st
  show Option.bind (f st) (fun st' => some st') = f st
  cases f st <;> rfl

theorem kcomp_assoc (f g h : Store → Option Store) :
    kcomp (kcomp f g) h = kcomp f (kcomp g h) := funext fun st => Option.bind_assoc _ _ _

theorem execLoopIters_zero (f : Env → Store → Option Store) (E : Env) (v : VarId) (lo : Int) :
    execLoopIters f E v lo 0 = (fun st => some st) :=
  funext fun st => rfl

theorem execLoopIters_succ (f : Env → Store → Option Store) (E : Env) (v : VarId) (lo : Int)
    (n : Nat) :
    execLoopIters f E v lo (n + 1) = kcomp (f (Env.set E v lo)) (execLoopIters f E v (lo + 1) n) :=
  funext fun st => rfl

/-- Function-level version of `execLoopIters_append`. -/
theorem execLoopIters_append' (f : Env → Store → Option Store) (env : Env) (v : VarId)
    (a b : Nat) (lo : Int) :
    execLoopIters f env v lo (a + b) =
    kcomp (execLoopIters f env v lo a) (execLoopIters f env v (lo + (a : Int)) b) :=
  funext fun st => execLoopIters_append f env v a b lo st

/-! ## Commutation (used by reorder) -/

/-- A single store transformer `X` commutes past a whole fold, provided it
    commutes with every step. -/
theorem execLoopIters_comm_left (f : Env → Store → Option Store) (E : Env) (v : VarId)
    (X : Store → Option Store)
    (h : ∀ (c : Int) (st : Store),
      kcomp (f (Env.set E v c)) X st = kcomp X (f (Env.set E v c)) st) :
    ∀ (n : Nat) (lo : Int),
      kcomp X (execLoopIters f E v lo n) = kcomp (execLoopIters f E v lo n) X := by
  intro n lo
  induction n generalizing lo with
  | zero => rw [execLoopIters_zero, kcomp_pure_right, kcomp_pure_left]
  | succ n ih =>
      rw [execLoopIters_succ, kcomp_assoc, ← kcomp_assoc, (funext fun st => h lo st).symm,
        kcomp_assoc, ih (lo + 1)]

/-- If every step of one fold commutes with every step of another, the two
    folds commute. -/
theorem execLoopIters_comm_folds (f g : Env → Store → Option Store) (E1 E2 : Env) (v : VarId)
    (h : ∀ (c1 c2 : Int) (st : Store),
      kcomp (f (Env.set E1 v c1)) (g (Env.set E2 v c2)) st =
      kcomp (g (Env.set E2 v c2)) (f (Env.set E1 v c1)) st) :
    ∀ (n1 n2 : Nat) (lo1 lo2 : Int),
      kcomp (execLoopIters f E1 v lo1 n1) (execLoopIters g E2 v lo2 n2) =
      kcomp (execLoopIters g E2 v lo2 n2) (execLoopIters f E1 v lo1 n1) := by
  intro n1 n2 lo1 lo2
  induction n2 generalizing lo2 with
  | zero => rw [execLoopIters_zero, kcomp_pure_right, kcomp_pure_left]
  | succ n2 ih =>
      have hcomm : kcomp (execLoopIters f E1 v lo1 n1) (g (Env.set E2 v lo2)) =
          kcomp (g (Env.set E2 v lo2)) (execLoopIters f E1 v lo1 n1) :=
        (execLoopIters_comm_left f E1 v _ (fun c st' => h c lo2 st') n1 lo1).symm
      rw [execLoopIters_succ, ← kcomp_assoc, hcomm, kcomp_assoc, ih (lo2 + 1), ← kcomp_assoc]

/-- Push a transformer `X` that commutes with every `A`-step past the first
    factor of a Kleisli composition. -/
theorem execLoopIters_push (A : Env → Store → Option Store) (E : Env) (v : VarId)
    (X G : Store → Option Store)
    (h : ∀ (c : Int) (st : Store),
      kcomp (A (Env.set E v c)) X st = kcomp X (A (Env.set E v c)) st) :
    ∀ (n : Nat) (lo : Int),
      kcomp X (kcomp (execLoopIters A E v lo n) G) =
      kcomp (execLoopIters A E v lo n) (kcomp X G) := by
  intro n lo
  induction n generalizing lo with
  | zero => rw [execLoopIters_zero, kcomp_pure_left, kcomp_pure_left]
  | succ n ih =>
      conv => lhs; rw [execLoopIters_succ, kcomp_assoc, ← kcomp_assoc,
        (funext fun st => h lo st).symm, kcomp_assoc, ih (lo + 1)]
      rw [execLoopIters_succ, kcomp_assoc]

/-- Interleaving `A`-then-`B` steps equals all `A` steps followed by all `B`
    steps, provided the `A` and `B` steps pairwise commute. -/
theorem execLoopIters_interleave (A B : Env → Store → Option Store) (env : Env) (v : VarId)
    (h : ∀ (j j' : Int) (st : Store),
      kcomp (A (Env.set env v j)) (B (Env.set env v j')) st =
      kcomp (B (Env.set env v j')) (A (Env.set env v j)) st) :
    ∀ (n : Nat) (lo : Int),
      execLoopIters (fun e st => kcomp (A e) (B e) st) env v lo n =
      kcomp (execLoopIters A env v lo n) (execLoopIters B env v lo n) := by
  intro n lo
  induction n generalizing lo with
  | zero =>
      rw [execLoopIters_zero, execLoopIters_zero, execLoopIters_zero]
      rfl
  | succ n ih =>
      rw [execLoopIters_succ, execLoopIters_succ, execLoopIters_succ, ih (lo + 1),
        kcomp_assoc,
        execLoopIters_push A env v (B (Env.set env v lo)) (execLoopIters B env v (lo + 1) n)
          (fun c st => h c lo st) n (lo + 1),
        ← kcomp_assoc]

/-- **Loop permutation (reorder).** Iterating an `N × M` rectangle row-major
    equals iterating it column-major, provided the body steps pairwise
    commute. This is the core of the `reorder` tactic. -/
theorem execLoopIters_swap (f : Env → Store → Option Store) (env : Env) (u v : VarId)
    (huv : u ≠ v) (M : Nat)
    (hcomm : ∀ (i j i' j' : Int) (st : Store),
      kcomp (f (Env.set (Env.set env u i) v j)) (f (Env.set (Env.set env u i') v j')) st =
      kcomp (f (Env.set (Env.set env u i') v j')) (f (Env.set (Env.set env u i) v j)) st) :
    ∀ (N : Nat) (s1 s2 : Int),
      execLoopIters (fun e st => execLoopIters f e v s2 M st) env u s1 N =
      execLoopIters (fun e st => execLoopIters f e u s1 N st) env v s2 M := by
  intro N s1 s2
  induction N generalizing s1 with
  | zero =>
      rw [execLoopIters_zero]
      have hz : ∀ e : Env, (fun st => execLoopIters f e u s1 0 st) = (fun st => some st) :=
        fun e => execLoopIters_zero f e u s1
      simp only [hz]
      rw [funext fun st => execLoopIters_pure env v M s2 st]
  | succ N ih =>
      -- split the outer fold into the first N u-iterations plus the last one
      rw [execLoopIters_append' (fun e st => execLoopIters f e v s2 M st) env u N 1 s1]
      -- the inner N+1 fold splits into N iterations then one; the v-fold
      -- interleaves A-then-B into all-A then all-B
      have hsplit : (fun e st => execLoopIters f e u s1 (N + 1) st)
          = (fun e st => kcomp (fun s => execLoopIters f e u s1 N s)
              (fun s => execLoopIters f e u (s1 + (N : Int)) 1 s) st) := by
        funext e st
        rw [execLoopIters_append' f e u N 1 s1]
      have hcommAB : ∀ (j j' : Int) (st : Store),
          kcomp (execLoopIters f (Env.set env v j) u s1 N)
                (execLoopIters f (Env.set env v j') u (s1 + (N : Int)) 1) st =
          kcomp (execLoopIters f (Env.set env v j') u (s1 + (N : Int)) 1)
                (execLoopIters f (Env.set env v j) u s1 N) st := by
        intro j j' st
        exact congrFun (execLoopIters_comm_folds f f (Env.set env v j) (Env.set env v j') u
          (fun c1 c2 st'' => by
            have hc := hcomm c1 j c2 j' st''
            simp only [Env.set_set_comm huv] at hc
            exact hc) N 1 s1 (s1 + (N : Int))) st
      rw [hsplit, execLoopIters_interleave _ _ env v hcommAB M s2, ih s1]
      -- the two remainders agree: one u-step vs. a v-fold of single u-steps
      have hrem : execLoopIters (fun e st => execLoopIters f e v s2 M st) env u (s1 + (N : Int)) 1
          = execLoopIters
              (fun e st => execLoopIters f e u (s1 + (N : Int)) 1 st) env v s2 M := by
        rw [execLoopIters_succ, execLoopIters_zero, kcomp_pure_right]
        refine funext fun st =>
          execLoopIters_congr f _ (Env.set env u (s1 + (N : Int))) env v ?_ M s2 st
        intro c st'
        show f (Env.set (Env.set env u (s1 + (N : Int))) v c) st' =
             execLoopIters f (Env.set env v c) u (s1 + (N : Int)) 1 st'
        rw [execLoopIters_succ, execLoopIters_zero, kcomp_pure_right,
          Env.set_set_comm (Ne.symm huv)]
      rw [hrem]

theorem bind_some_pure (x : Option Store) : Option.bind x (fun y => some y) = x := by
  cases x <;> rfl

/-- Generalized substitution folding: the fold body `f1` runs with `v` bound to
    `g env'` (an env-dependent value), while `f2` is the pre-substituted body.
    `g` must be insensitive to the loop variable's own binding — which holds
    exactly when the substitution expression does not mention the loop variable. -/
theorem execLoopIters_subst_gen (f1 f2 : Env → Store → Option Store) (env : Env)
    (v lv : VarId) (g : Env → Int) (lo : Int) (iters : Nat) (store : Store)
    (hlv : lv ≠ v)
    (hg : ∀ (env' : Env) (x : Int), g (Env.set env' lv x) = g env')
    (h : ∀ (env' : Env) (st : Store), f1 (Env.set env' v (g env')) st = f2 env' st) :
    execLoopIters f1 (Env.set env v (g env)) lv lo iters store =
    execLoopIters f2 env lv lo iters store := by
  induction iters generalizing lo store with
  | zero => rfl
  | succ n ih =>
      simp only [execLoopIters]
      have hfirst : f1 (Env.set (Env.set env v (g env)) lv lo) store
          = f2 (Env.set env lv lo) store := by
        rw [Env.set_set_comm (Ne.symm hlv), ← hg env lo]
        exact h (Env.set env lv lo) store
      rw [hfirst]
      congr 1
      funext st'
      exact ih (lo + 1) st'

/-! ## Blocking (tile) -/

/-- One outer iteration of a tiled loop (an inner fold over `lv2`) equals a
    contiguous block of `m` iterations of the flat loop, where the flat loop
    variable is set to `t + j` for inner iteration `j`. -/
theorem execLoopIters_block (F f : Env → Store → Option Store) (env : Env)
    (v lv1 lv2 : VarId) (q : Int) :
    ∀ (m : Nat) (t : Int) (store : Store),
      (∀ (j : Int) (st : Store),
        F (Env.set (Env.set env lv1 q) lv2 j) st = f (Env.set env v (t + j)) st) →
      execLoopIters F (Env.set env lv1 q) lv2 0 m store =
      execLoopIters f env v t m store := by
  intro m
  induction m with
  | zero => intro t store _; rfl
  | succ m ih =>
      intro t store hstep
      rw [execLoopIters_append F (Env.set env lv1 q) lv2 m 1 0 store,
        execLoopIters_append f env v m 1 t store, ih t store hstep]
      have hz : (0 : Int) + (m : Int) = (m : Int) := by push_cast; omega
      simp only [execLoopIters_succ, execLoopIters_zero, kcomp_pure_right, hz]
      congr 1
      funext a
      exact hstep (m : Int) a

/-- **Loop partition (tile).** A flat fold of `m * n` iterations equals a nested
    fold: `n` outer iterations, each running an inner fold of `m` iterations,
    with the flat variable reconstructed as `q * m + j`. -/
theorem execLoopIters_partition (F f : Env → Store → Option Store) (env : Env)
    (v lv1 lv2 : VarId) (m : Nat)
    (hstep : ∀ (q j : Int) (st : Store),
      F (Env.set (Env.set env lv1 q) lv2 j) st = f (Env.set env v (q * (m : Int) + j)) st) :
    ∀ (n : Nat) (s : Int) (store : Store),
      execLoopIters (fun e st => execLoopIters F e lv2 0 m st) env lv1 s n store =
      execLoopIters f env v (s * (m : Int)) (m * n) store := by
  intro n
  induction n with
  | zero => intro s store; rfl
  | succ n ih =>
      intro s store
      rw [Nat.mul_succ, execLoopIters_append (fun e st => execLoopIters F e lv2 0 m st)
            env lv1 n 1 s store,
        execLoopIters_append f env v (m * n) m (s * (m : Int)) store, ih s store]
      have hstep1 : execLoopIters (fun e st => execLoopIters F e lv2 0 m st)
          env lv1 (s + (n : Int)) 1
          = execLoopIters F (Env.set env lv1 (s + (n : Int))) lv2 0 m := by
        funext store
        rw [execLoopIters_succ, execLoopIters_zero, kcomp_pure_right]
      have hblk : execLoopIters F (Env.set env lv1 (s + (n : Int))) lv2 0 m
          = execLoopIters f env v ((s + (n : Int)) * (m : Int)) m := by
        funext store
        exact execLoopIters_block F f env v lv1 lv2 (s + (n : Int)) m
          ((s + (n : Int)) * (m : Int)) store (fun j st => hstep (s + (n : Int)) j st)
      rw [hstep1, hblk]
      have hoff : (s + (n : Int)) * (m : Int) = (s * (m : Int)) + ((m * n : Nat) : Int) := by
        push_cast; ring
      rw [hoff]

/-! ## Fusion -/

/-- One fused iteration block: an inner fold of `M` iterations over `v2` (with
    `v1` fixed at `q`) equals `M` iterations of the fused loop over `w`, where
    fused iteration `j` corresponds to `w = t + j`. -/
theorem execLoopIters_fuse_block (FF F : Env → Store → Option Store) (env : Env)
    (w v1 v2 : VarId) (q : Int) :
    ∀ (M : Nat) (t : Int) (store : Store),
      (∀ (j : Int) (st : Store), 0 ≤ j → j < (M : Int) →
        FF (Env.set env w (t + j)) st = F (Env.set (Env.set env v1 q) v2 j) st) →
      execLoopIters F (Env.set env v1 q) v2 0 M store =
      execLoopIters FF env w t M store := by
  intro M
  induction M with
  | zero => intro t store _; rfl
  | succ m ih =>
      intro t store hstep
      rw [execLoopIters_append F (Env.set env v1 q) v2 m 1 0 store,
        execLoopIters_append FF env w m 1 t store,
        ih t store (fun j st hj0 hjlt => hstep j st hj0 (by omega))]
      have hz : (0 : Int) + (m : Int) = (m : Int) := by push_cast; omega
      have hm0 : (0 : Int) ≤ (↑m : Int) := by omega
      have hmlt : (↑m : Int) < ((m + 1 : Nat) : Int) := by omega
      simp only [execLoopIters_succ, execLoopIters_zero, kcomp_pure_right, hz]
      congr 1
      funext a
      exact (hstep (m : Int) a hm0 hmlt).symm

/-- **Loop fusion.** Two nested loops (`v1` over `N` iterations, `v2` over `M`)
    equal a single flat loop over `w` with `M * N` iterations, where fused
    iteration `q * M + j` corresponds to `(v1, v2) = (q, j)`. -/
theorem execLoopIters_fuse (FF F : Env → Store → Option Store) (env : Env)
    (w v1 v2 : VarId) (M : Nat)
    (hstep : ∀ (q j : Int) (st : Store), 0 ≤ j → j < (M : Int) →
      FF (Env.set env w (q * (M : Int) + j)) st = F (Env.set (Env.set env v1 q) v2 j) st) :
    ∀ (N : Nat) (a : Int) (store : Store),
      execLoopIters (fun e st => execLoopIters F e v2 0 M st) env v1 a N store =
      execLoopIters FF env w (a * (M : Int)) (M * N) store := by
  intro N
  induction N with
  | zero => intro a store; rfl
  | succ n ih =>
      intro a store
      rw [Nat.mul_succ, execLoopIters_append (fun e st => execLoopIters F e v2 0 M st)
            env v1 n 1 a store,
        execLoopIters_append FF env w (M * n) M (a * (M : Int)) store, ih a store]
      have hstep1 : execLoopIters (fun e st => execLoopIters F e v2 0 M st)
          env v1 (a + (n : Int)) 1
          = execLoopIters F (Env.set env v1 (a + (n : Int))) v2 0 M := by
        funext store
        rw [execLoopIters_succ, execLoopIters_zero, kcomp_pure_right]
      have hblk : execLoopIters F (Env.set env v1 (a + (n : Int))) v2 0 M
          = execLoopIters FF env w ((a + (n : Int)) * (M : Int)) M := by
        funext store
        exact execLoopIters_fuse_block FF F env w v1 v2 (a + (n : Int)) M
          ((a + (n : Int)) * (M : Int)) store
          (fun j st hj0 hjlt => hstep (a + (n : Int)) j st hj0 hjlt)
      rw [hstep1, hblk]
      have hoff : (a + (n : Int)) * (M : Int) = (a * (M : Int)) + ((M * n : Nat) : Int) := by
        push_cast; ring
      rw [hoff]

end VeriTac
