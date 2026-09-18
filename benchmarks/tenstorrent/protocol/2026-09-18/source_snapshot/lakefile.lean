import Lake
open Lake DSL

package VeriTac where
  leanOptions := #[⟨`autoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"

@[default_target]
lean_lib VeriTac where
  roots := #[`VeriTac]

lean_exe veritac where
  root := `Main

lean_exe gemmini_check where
  root := `GemminiMain

lean_exe gemmini_program_check where
  root := `GemminiProgramMain
