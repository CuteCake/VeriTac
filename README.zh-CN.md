# VeriTac —— 面向智能体时代的 ML 编译器概念（探索阶段）

[English](README.md) | **简体中文**

一个**形式化验证的、基于策略（tactic）的** ML 编译器。张量优化被表达为经过 Lean 4 形式化验证的策略，AI 搜索智能体提出的策略序列**构造即正确**（correct by construction）。

本仓库与许多其他「智能体写内核」仓库的本质区别在于：它们需要一个厂商提供的编译器（如 CUDA）来验证代码正确性。但如果**你自己就是 ASIC 厂商，没有任何可参照的实现**呢？本仓库保证两点：1. 优化的每一步都是正确的；2. 优化的正确性可以由 Lean 自动验证。当这套体系完成时，它将成为 ML 编译器工程的终极形态。

### 端到端 GEMM 演示

`examples/gemm_demo.py` 在一个方阵矩阵乘法上跑完整流水线：

1. 搜索智能体提出调度方案（`tile` / `fuse` / `reorder` / `parallel` / `vectorize`），**Lean CLI 对每个策略重新验证**。
2. 调度被降级为 C 代码。OpenMP 编译指示（`#pragma omp parallel for`、`#pragma omp simd`）**仅当**写依赖检查（镜像已证明的 `indexLocalP_flat_leading` 判据）确认循环相互独立时才生成——对 `k` 的归约循环会正确地得到注释而非编译指示。
3. C 代码用支持 OpenMP 的编译器编译，并与 numpy 对比基准测试。

在本机实测（gcc-15，10 线程）：智能体的最佳调度（`parallel i0`、`vectorize i2`、`parallel i1`）在 128³ 下达到 **4.5x** 加速；完整寄存器分块链（`tile(32)`×2 → `reorder i1_inner i2` → `vectorize i1_inner` → `parallel i0_outer`）在 256³ 下达到 **12.8x**（18.9 ms → 1.47 ms），且与 numpy 结果完全一致。Lean 证明与代码生成层的防护共同保证了被标注的循环确实可以安全并行化。

有两个演示入口：

- `examples/gemm_demo.py [dim]` —— 离线流水线：搜索智能体 → Lean CLI → C → 基准测试。
- `examples/llm_gemm_demo.py` —— 带完整轨迹输出的 LLM 驱动流水线。从 `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL`（或根目录 `.env`，或交互式提示）读取 API 密钥与模型；每一轮由一个 LLM 提出一个策略，Lean CLI 验证并应用之，轨迹打印所提出的策略、结果语句树以及任何被拒绝的信息（并作为反馈回传给模型），最后对最终调度做基准测试。`--mock` 用固定调度运行同一轨迹，无需 API 密钥。

## 总览

VeriTac 把**算什么**（语义 IR）与**怎么算**（调度）分离开来。优化被实现为可组合、已验证的策略，它们在保持语义的同时变换循环嵌套。AI 搜索智能体提出策略序列，验证器保证每个被接受的序列都产生正确代码。

```
用户程序 → 语义 IR → [已验证策略] → 调度后 IR → C 代码
                            ↑
                     AI 搜索智能体
```

## 项目结构

```
VeriTac/
├── lakefile.lean                 # Lake 构建配置（Lean 4 + Mathlib）
├── lean-toolchain                # Lean 4.29.0-rc6
├── Main.lean                     # CLI：JSON 进 / JSON 出的策略应用
├── VeriTac/
│   ├── Basic.lean                # 再导出所有模块
│   ├── IR/
│   │   ├── Shape.lean            # Shape（List Nat）、Index（依赖类型 Fin 向量）
│   │   ├── TExpr.lean            # 语义 IR：const、tensor、map、zip、reduce
│   │   ├── Denote.lean           # 指称语义（TExpr → Index → α）
│   │   └── Operations.lean       # tensorAdd、tensorMap、tensorSum
│   ├── Schedule/
│   │   ├── LoopNest.lean         # Stmt、SExpr、Env、Store、execStmt（基于 fuel）
│   │   ├── Equiv.lean            # ScheduleEquiv：∀ env store, exec s1 = exec s2
│   │   └── Lower.lean            # TExpr → LoopNest（朴素嵌套循环）
│   ├── Tactic/
│   │   ├── Tile.lean             # 把循环拆成 outer/inner 两层
│   │   ├── Split.lean            # tile 的别名
│   │   ├── Fuse.lean             # 通过 div/mod 合并相邻循环
│   │   ├── Reorder.lean          # 交换相互独立的循环
│   │   ├── Unroll.lean           # 完全展开常量边界的循环
│   │   ├── Vectorize.lean        # 给最内层循环标注 SIMD
│   │   ├── Parallel.lean         # 给循环标注并行执行
│   │   ├── CacheRead.lean        # 插入本地缓冲 + 拷贝 + 读取替换
│   │   └── Library.lean          # 策略注册表与 applyTactic 分发器
│   ├── Compose/
│   │   ├── Precondition.lean     # 按策略种类做前置条件检查
│   │   └── Engine.lean           # applySchedule：顺序策略组合
│   └── Util/
│       ├── Finset.lean           # 求和划分引理（用于 tiling 证明）
│       └── List.lean             # List 交换工具
├── CodeGen/                      # Python —— 未被验证的 C 代码生成器
│   ├── lower.py                  # 解析 LoopNest JSON → Python AST
│   ├── emit_c.py                 # Python AST → 带 OpenMP 编译指示的 C 代码
│   └── runner.py                 # 编译、运行、与 numpy 基准对比
├── Search/                       # Python —— 暴力搜索智能体
│   ├── interface.py              # Lean CLI 子进程封装
│   ├── cost_model.py             # 解析式成本模型（运算 + 内存流量）
│   └── agent.py                  # 枚举策略序列，按成本排序
└── tests/
    └── test_matmul.py            # 端到端矩阵乘法测试
```

## 构建

### 前置依赖

- [Lean 4](https://leanprover.github.io/lean4/doc/setup.html)（通过 `elan` 安装）
- Python 3.9+
- `clang`（用于编译 C 代码）
- `numpy`（用于基准对比）

### 构建 Lean 工程

```bash
lake update    # 获取 Mathlib（首次会下载预构建缓存，约 5 分钟）
lake build     # 构建库（534 个模块）
lake build veritac  # 构建 CLI 二进制
```

### 验证

```bash
# 运行端到端 matmul（代码生成）测试
PYTHONPATH=. python3 tests/test_matmul.py

# 运行健全性回归测试（tile 拒绝、unroll、标注）
PYTHONPATH=. python3 tests/test_soundness.py
```

## 使用

### CLI

`veritac` 二进制从 stdin 或 `--json` 读取 JSON，对一个循环嵌套语句应用一系列策略。

```bash
# 对简单循环应用 tile(i0, 32)
.lake/build/bin/veritac --json '{
  "stmt": {
    "tag": "loop", "var": "i0",
    "lo": {"tag": "lit", "val": 0},
    "hi": {"tag": "lit", "val": 256},
    "ann": "none",
    "body": {
      "tag": "bufWrite", "buf": "C",
      "indices": [{"tag": "var", "name": "i0"}],
      "val": {"tag": "bufRead", "buf": "A",
              "indices": [{"tag": "var", "name": "i0"}]}
    }
  },
  "tactics": [
    {"kind": "tile", "vars": ["i0"], "int_params": [32]},
    {"kind": "parallel", "vars": ["i0_outer"]}
  ]
}'
```

**输出：**

```json
{
  "applied": 2,
  "stmt": {
    "tag": "loop", "var": "i0_outer", "ann": "parallel",
    "body": {
      "tag": "loop", "var": "i0_inner", "ann": "none",
      "body": { "..." }
    }
  }
}
```

### JSON 格式

#### 语句（`Stmt`）

| Tag | 字段 | 说明 |
|-----|--------|-------------|
| `skip` | — | 空操作 |
| `bufWrite` | `buf`、`indices`、`val` | 向缓冲写入值 |
| `loop` | `var`、`lo`、`hi`、`ann`、`body` | 带标注的循环 |
| `seq` | `s1`、`s2` | 顺序组合 |
| `alloc` | `buf`、`shape`、`body` | 分配本地缓冲 |

#### 表达式（`SExpr`）

| Tag | 字段 | 说明 |
|-----|--------|-------------|
| `lit` | `val`（int） | 整数字面量 |
| `var` | `name`（string） | 变量引用 |
| `add`、`mul`、`div`、`mod` | `left`、`right` | 二元算术 |
| `bufRead` | `buf`、`indices` | 从缓冲读取 |

#### 标注

`"none"`、`"parallel"`、`"vectorize"`、`"unrolled"`

#### 策略应用

```json
{
  "kind": "<tactic_name>",
  "vars": ["<target_var>", ...],
  "int_params": [<nat>, ...],
  "str_params": ["<string>", ...]
}
```

### 可用策略

| 策略 | vars | int_params | 说明 |
|--------|------|------------|-------------|
| `tile` | `[var]` | `[tile_size]` | 把循环拆成 outer/inner |
| `split` | `[var]` | `[factor]` | tile 的别名 |
| `fuse` | `[var1, var2]` | — | 合并两个相邻的顺序循环 |
| `reorder` | `[var1, var2]` | — | 交换两个相邻的独立循环 |
| `unroll` | `[var]` | — | 完全展开循环（要求常量边界） |
| `vectorize` | `[var]` | — | 给最内层循环标注 SIMD |
| `parallel` | `[var]` | — | 给循环标注并行执行 |
| `cache_read` | — | — | 插入本地缓存缓冲（进阶） |

### Python 代码生成

```python
import json
from CodeGen.lower import parse_stmt
from CodeGen.emit_c import emit_function

# 从 Lean CLI 获取优化后的循环嵌套
result = json.loads(subprocess.check_output([
    ".lake/build/bin/veritac", "--json", json.dumps({
        "stmt": matmul_stmt,
        "tactics": [
            {"kind": "tile", "vars": ["i0"], "int_params": [32]},
            {"kind": "parallel", "vars": ["i0_outer"]}
        ]
    })
]))

# 生成 C 代码
stmt = parse_stmt(result["stmt"])
c_code = emit_function("matmul", stmt,
    input_bufs=[("A", M*K), ("B", K*N)],
    output_bufs=[("C", M*N)])
```

### 搜索智能体

```bash
# 为 matmul 寻找最优策略序列
PYTHONPATH=. python3 -m Search.agent --max-length 3 --top-k 5
```

该智能体的工作流程：

1. 枚举给定长度以内的策略序列
2. 通过 Lean CLI 验证每一个（前置条件检查）
3. 用解析式成本模型为有效序列打分
4. 输出 Top-K 结果

## 架构

### 双层 IR

**语义 IR（`TExpr`）**描述**算什么**：

- 以 `α`（任意类型）为参数——对所有类型都成立，用 `Nat`/`Int` 测试
- `denote : TExpr α s → (Index s → α)` 给出数学含义
- 已证明引理：`denote (map g (map f e)) = denote (map (g ∘ f) e)`

**调度 IR（`Stmt`/`SExpr`）**描述**怎么算**：

- 带显式迭代、缓冲读写与标注的循环嵌套
- 用于并行与向量化的标注
- 基于 fuel 的 `execStmt` 语义以保证可终止性

### 正确性

每个策略都有一个形如下的正确性定理：

```
theorem tactic_correct : original_stmt ≈ₛ transformed_stmt
```

其中 `s1 ≈ₛ s2` 表示 `∀ fuel env store, execStmt fuel env store s1 = execStmt fuel env store s2`。

组合正确性由传递性导出：

```lean
theorem ScheduleEquiv.trans : s1 ≈ₛ s2 → s2 ≈ₛ s3 → s1 ≈ₛ s3
```

### 证明状态

验证框架已搭建完成，几个核心正确性定理现已在机器上校验通过（无 `sorry`）。以下是**已证明**的：

| 定理 | 状态 | 说明 |
|---------|--------|-------|
| `vectorize_correct`、`parallel_correct` | ✅ 已证明 | `execStmt` 忽略循环标注，因此在模型中它们是语法层面的空操作 |
| `substExprVar_eval` | ✅ 已证明 | 表达式级替换（`v ↦ e` 在 `env[v := e]` 下求值） |
| `substStmtVar_lit_correct` | ✅ 已证明 | 对**字面量** `v ↦ lit k` 的语句级替换 |
| `unroll_correct` | ✅ 已证明 | 通过 `unrollBody_correct`，把 `execLoopIters` 与展开后的语句联系起来 |
| `tile_correct` | ✅ 已证明 | 通过 `execLoopIters_partition` + `substStmtVar_correct` |
| `fuse_correct` | ✅ 已证明 | 通过 `execLoopIters_fuse` + `substStmtVar_correct` |
| `reorder_correct` | ✅ 已证明 | 以语义交换假设重述 |
| `cacheRead_correct` | ✅ 已证明 | 重述：从镜像 store 中读取替换是空操作 |
| `split_correct` | ✅ 已证明 | 归结为 `tile_correct` |

所有策略正确性定理现在都带有**零 `sorry` 的机器校验证明**。证明基础设施位于 `VeriTac/Schedule/`：

| 基础设施 | 提供什么 |
|----------------|------------------|
| `LoopComposition.lean` | 函数级 Kleisli 组合（`kcomp`）、块**划分**（tile）、**交换**（reorder）、**合并**（fuse），以及关于 `execLoopIters` 的交错/交换与一般替换引理 |
| `FreeVars.lean` | `varFreeStmt`（考虑边界的捕获分析）、`exprStatic`、只读/读集谓词 |
| `ExecLemmas.lean` | 解释器的总体性、在未使用绑定下 store 的不变性、只读与写集纪律、「脱离缓冲达成一致」（blind）引理 |

证明本身：

- **`tile_correct`** —— 通过 `execLoopIters_partition` + 一般语句替换引理 `substStmtVar_correct`（假设：`loopBinds v body = false`、生成的 `_outer`/`_inner` 名字是新的、且 tile 大小整除循环范围）。`tileAt` 的外层上界现在是精确商 `hi/ts`。
- **`fuse_correct`** —— 通过 `execLoopIters_fuse`，使用 `(i, j) ↔ i*M + j` 双射并满足 `0 ≤ j < M`，以及两次应用 `substStmtVar_correct`（合并后的名字必须对 body 是新的）。
- **`reorder_correct`** —— 诚实地重述：它要求 `v1 ≠ v2`、边界与另一轴无关（`varInExpr`/`exprStatic` 无关）**并且**对 body 存在语义上的两两交换假设。Lean 定理精确编码了为什么 `loopsIndependent` 只是语法启发式；为具体 body 完成交换假设的证明是依赖分析的责任。
- **`cacheRead_correct`** —— 诚实地重述：当缓存缓冲在起始 store 中镜像了源缓冲、且两个缓冲对语句都是只读（`isReadOnly`）时，把 `origBuf ↦ cacheBuf` 的读取替换是空操作。`cacheRead` 插入的*拷贝循环*必须建立这种镜像关系，这是（Python 侧的）剩余责任。
- **并行依赖桥接** —— `indexLocalP_flat_leading` 证明：当循环体每个写索引对该轴变量都是*局部的*（以裸 `v` 作为领头的索引，其余部分与 `v` 无关且与 store 无关）时，不同的轴值写入不相交的扁平单元，因此各迭代是两两独立的。这是 `parallel` 标注背后健全的依赖判据（`loopLocalWritesP`）；语法上的 `checkIndependence` 仍是搜索侧的启发式。
- **`split_correct`** —— 归结为 `tile_correct`（同一定理，现已证明）。

#### 所做的健全性修复

在审计语义时发现并修复了若干真正的健全性缺陷：

1. **fuel 在每个循环嵌套层都被消耗。** `execStmt` 在进入循环体时会递减 fuel，导致重构嵌套（分块会加一层）会耗尽叶计算的预算，使 `tile_correct` 的 `∀ fuel` 陈述为假。现在 fuel 被携带但不再消耗：每个循环都有有限的迭代次数，因此解释器是整体的，分块也不会改变叶层获得的预算。
2. **当 tile 大小不能整除范围时 `tile` 会越界。** 内层循环上界被硬编码为 `tileSize`，导致当 `hi % tileSize ≠ 0` 时最后一个块迭代越过了 `hi`。现在 `tile` 要求字面量 `[0, hiVal)` 范围且 `hiVal % tileSize = 0`，否则拒绝。
3. **`fuse` 用 div/mod 拆分对顺序循环做合并**，这会把第二个 body 运行 `N₁ * N₂` 次。现在正确地合并*嵌套*循环。

> **关于标注的说明。** `execStmt` 忽略循环标注，因此 `parallel` 与 `vectorize` 在 Lean 模型中是平凡地「正确」的。生成代码中并行/SIMD 的*真正*正确性仍需对标注做真实的依赖分析（在带循环携带依赖的循环上做 `parallel` 会生成错误的 `#pragma omp parallel` 代码）。Lean 证明建立了循环嵌套的语义等价；代码生成层的依赖属性是另一项待完成的责任。

## 设计决策

1. **数值抽象**：IR 以任意类型为参数；浮点语义推迟。
2. **Shape 用 `List Nat`**：避免繁重的依赖类型管道。
3. **fuel 携带但不消耗**：循环边界有限，解释器是整体的。fuel 保留在签名中但从不减少，这使得循环重构等价定理可以对任意 `fuel` 值证明。
4. **Lean-Python 接口**：通过子进程走 JSON（无需 FFI）。
5. **步长 1000 的扁平索引**：Lean 执行与 C 代码生成使用相同的约定以保证正确性对齐。
6. **选择性使用 Mathlib**：`Finset`、`Fin`、`omega` —— 预构建缓存，加快构建。

## 许可证

MIT
