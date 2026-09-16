import VeriTac.Gemmini.Execution16
import VeriTac.Gemmini.K32Spec

/-!
Execution certificate for the K=32 reuse_b schedule (M=32, N=16, K=32).

`reuseK32` mirrors, instruction for instruction, the command stream printed by
`CodeGen.gemmini.gen_reuse_b` for `{'m': 32, 'n': 16, 'k': 32, 'schedule':
'reuse_b'}`:

    config_ex {'dataflow': 1}
    config_st {'stride_bytes': 64}
    config_ld {'slot': 0, 'stride_bytes': 32, 'scale': 1.0}
    config_ld {'slot': 1, 'stride_bytes': 16, 'scale': 1.0}
    mvin {'slot': 1, 'buf': 'B', 'offset': 0, 'spad_addr': 16}
    mvin {'slot': 0, 'buf': 'A', 'offset': 0, 'spad_addr': 0}
    preload {'bd_spad_addr': 16, 'out_addr': 2684354560}
    compute {'accumulated': False, 'a_spad_addr': 0}
    mvin {'slot': 0, 'buf': 'A', 'offset': 512, 'spad_addr': 0}
    preload {'bd_spad_addr': 4294967295, 'out_addr': 2684354576}
    compute {'accumulated': True, 'a_spad_addr': 0}
    mvin {'slot': 1, 'buf': 'B', 'offset': 256, 'spad_addr': 16}
    mvin {'slot': 0, 'buf': 'A', 'offset': 16, 'spad_addr': 0}
    preload {'bd_spad_addr': 16, 'out_addr': 3758096384}
    compute {'accumulated': False, 'a_spad_addr': 0}
    mvout {'buf_offset': 0, 'acc_addr': 3758096384}
    mvin {'slot': 0, 'buf': 'A', 'offset': 528, 'spad_addr': 0}
    preload {'bd_spad_addr': 4294967295, 'out_addr': 3758096400}
    compute {'accumulated': True, 'a_spad_addr': 0}
    mvout {'buf_offset': 256, 'acc_addr': 3758096400}
    fence {}

Two reuse effects are exercised: the B tile loaded for a K tile stays latched
in `weights` and is consumed by the second output-row tile's retained compute,
and the accumulator keeps the nonzero kk0 partials (addresses `0xa0000000`
overwrite / `0xa0000010` second row) which kk1 accumulates onto (addresses
`0xe0000000` / `0xe0000010`).
-/
namespace VeriTac.Gemmini

def reuseK32 : Program := [
  .configEx 1, .configSt 64, .configLd 0 32 true, .configLd 1 16 true,
  .mvin 1 .b 0 16 16 16, .mvin 0 .a 0 0 16 16,
  .preload 16 2684354560, .compute false 0 4294967295,
  .mvin 0 .a 512 0 16 16,
  .preload 4294967295 2684354576, .compute true 0 4294967295,
  .mvin 1 .b 256 16 16 16, .mvin 0 .a 16 0 16 16,
  .preload 16 3758096384, .compute false 0 4294967295,
  .mvout 0 3758096384 16 16,
  .mvin 0 .a 528 0 16 16,
  .preload 4294967295 3758096400, .compute true 0 4294967295,
  .mvout 256 3758096400 16 16, .fence]

set_option maxRecDepth 10000 in
set_option maxHeartbeats 1600000 in
theorem run_reuseK32_observe (p : GemminiPlan) (a b : Nat → Int)
    (hm : p.m = 32) (hn : p.n = 16) (hk : p.k = 32)
    (hsp : 32 ≤ p.scratchpadRows) (hac : 32 ≤ p.accumulatorRows)
    (r c : Nat) (hr : r < 32) (hc : c < 16) :
    observe (runProgram p a b reuseK32) (r * 16 + c) =
      some (k32Result a b r c, true, true) := by
  have hsp16 : 16 ≤ p.scratchpadRows := by omega
  have hac16 : 16 ≤ p.accumulatorRows := by omega
  have hdiv : (r * 16 + c) / 16 = r := by omega
  have hmod : (r * 16 + c) % 16 = c := by omega
  by_cases hlo : r < 16
  · have hncover : ¬ 256 ≤ r * 16 + c := by omega
    have hidxA1 : ∀ t : Nat, 16 + r * 32 + t = r * 32 + 16 + t := by omega
    have hidxB1 : ∀ t : Nat, 256 + t * 16 + c = (16 + t) * 16 + c := by omega
    simp [observe, runProgram, runFrom, reuseK32, step, initialState,
      tileFits, accAddress, accRow, accumulates, tileProduct, k32Result,
      hm, hn, hk, hsp, hsp16, hac, hac16, hdiv, hmod, hc, hlo, hncover,
      hidxA1, hidxB1]
  · have hcover : 256 ≤ r * 16 + c := by omega
    have hdiv2 : (r * 16 + c - 256) / 16 = r - 16 := by omega
    have hmod2 : (r * 16 + c - 256) % 16 = c := by omega
    have hr2 : r - 16 < 16 := by omega
    have hidxA0 : ∀ t : Nat, 512 + (r - 16) * 32 + t = r * 32 + t := by omega
    have hidxA1 : ∀ t : Nat, 528 + (r - 16) * 32 + t = r * 32 + 16 + t := by omega
    have hidxB1 : ∀ t : Nat, 256 + t * 16 + c = (16 + t) * 16 + c := by omega
    simp [observe, runProgram, runFrom, reuseK32, step, initialState,
      tileFits, accAddress, accRow, accumulates, tileProduct, k32Result,
      hm, hn, hk, hsp, hsp16, hac, hac16, hdiv, hmod, hc, hlo,
      hcover, hdiv2, hmod2, hr2, hidxA0, hidxA1, hidxB1]

end VeriTac.Gemmini
