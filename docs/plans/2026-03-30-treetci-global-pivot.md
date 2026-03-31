# TreeTCI Global Pivot Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** TreeTCI の optimize ループに global pivot search を追加し、局所的な特徴を持つ関数の収束を改善する。

**Architecture:** TCI2 の `DefaultGlobalPivotFinder` と同じ greedy local search アルゴリズムを TreeTCI 用に適応。`to_treetn` で materialization → `TreeTN::evaluate` で近似値を計算 → batch_eval で真値と比較。`global_search_interval` で実行頻度を制御。

**Tech Stack:** Rust, tensor4all-treetci crate, tensor4all-treetn (evaluate)

**Working directory:** `/home/shinaoka/tensor4all/tensor4all-rs`

**Reference:**
- TCI2 GlobalPivotFinder: `crates/tensor4all-tensorci/src/globalpivot.rs`
- TCI.jl テスト: `/home/shinaoka/tensor4all/TensorCrossInterpolation.jl/test/test_tensorci2.jl` L433-458

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `crates/tensor4all-treetci/src/optimize.rs` | TreeTciOptions にパラメータ追加 + ループに global search 組み込み |
| Create | `crates/tensor4all-treetci/src/globalpivot.rs` | TreeTCI 用 global pivot finder |
| Modify | `crates/tensor4all-treetci/src/lib.rs` | mod globalpivot + re-exports |
| Modify | `crates/tensor4all-treetci/src/optimize.rs` (tests) | テスト追加 |

---

## Task 1: TreeTciOptions にパラメータ追加

**Files:**
- Modify: `crates/tensor4all-treetci/src/optimize.rs`

- [ ] **Step 1: TreeTciOptions に global pivot パラメータを追加**

```rust
#[derive(Clone, Debug)]
pub struct TreeTciOptions {
    pub tolerance: f64,
    pub max_iter: usize,
    pub max_bond_dim: usize,
    pub normalize_error: bool,
    /// Run global pivot search every N iterations. 0 = disabled.
    pub global_search_interval: usize,
    /// Maximum number of global pivots to add per search.
    pub max_global_pivots: usize,
    /// Number of random starting points for global pivot search.
    pub num_global_searches: usize,
    /// Only add pivots where error > tolerance × this margin.
    pub global_search_tol_margin: f64,
}

impl Default for TreeTciOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 20,
            max_bond_dim: usize::MAX,
            normalize_error: true,
            global_search_interval: 0,  // disabled by default
            max_global_pivots: 5,
            num_global_searches: 5,
            global_search_tol_margin: 10.0,
        }
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build -p tensor4all-treetci --release 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```bash
git add crates/tensor4all-treetci/src/optimize.rs
git commit -m "feat(treetci): add global pivot search parameters to TreeTciOptions"
```

---

## Task 2: Global pivot finder モジュール

**Files:**
- Create: `crates/tensor4all-treetci/src/globalpivot.rs`
- Modify: `crates/tensor4all-treetci/src/lib.rs`

- [ ] **Step 1: Write failing test**

Append to `crates/tensor4all-treetci/src/globalpivot.rs` (at the end, inside `#[cfg(test)]` module):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TreeTciGraph, TreeTciEdge, SimpleTreeTci};

    #[test]
    fn test_find_global_pivots_finds_error_points() {
        // 3-site chain: 0-1-2, local_dims = [4, 4, 4]
        let graph = TreeTciGraph::new(3, &[
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
        ]).unwrap();
        let local_dims = vec![4, 4, 4];

        // Function with localized feature: large only when all indices > 2
        let f = |idx: &[usize]| -> f64 {
            if idx.iter().all(|&x| x >= 2) {
                100.0
            } else {
                (idx[0] as f64) * 0.1 + (idx[1] as f64) * 0.01
            }
        };

        // Zero approximation → evaluate always returns 0
        // So error = |f(x) - 0| = |f(x)|
        let approx_eval = |idx: &[usize]| -> f64 { 0.0 };

        let finder = DefaultTreeGlobalPivotFinder::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let pivots = finder.find_pivots(
            &local_dims,
            &f,
            &approx_eval,
            0.1,  // abs_tol: only accept error > 0.1 * 10.0 = 1.0
            &mut rng,
        );

        // Should find pivots in the high-value region (indices >= 2)
        assert!(!pivots.is_empty(), "Should find at least one global pivot");
        for pivot in &pivots {
            assert_eq!(pivot.len(), 3);
        }
    }
}
```

- [ ] **Step 2: Implement global pivot finder**

Create `crates/tensor4all-treetci/src/globalpivot.rs`:

```rust
//! Global pivot finder for TreeTCI.
//!
//! Adapted from TCI2's DefaultGlobalPivotFinder. Uses greedy local search
//! from random starting points to find multi-indices with large interpolation
//! error, which are then added as global pivots.

use rand::Rng;
use tensor4all_tcicore::MultiIndex;

/// Default global pivot finder for TreeTCI.
///
/// Algorithm (same as TCI2):
/// 1. Generate `num_searches` random initial points
/// 2. From each point, sweep all dimensions to find local error maximum
/// 3. Keep points where error > `abs_tol × tol_margin`
/// 4. Limit to `max_pivots` results
#[derive(Debug, Clone)]
pub struct DefaultTreeGlobalPivotFinder {
    /// Number of random starting points for greedy search.
    pub num_searches: usize,
    /// Maximum number of pivots to return per call.
    pub max_pivots: usize,
    /// Only return pivots where error > abs_tol × tol_margin.
    pub tol_margin: f64,
}

impl Default for DefaultTreeGlobalPivotFinder {
    fn default() -> Self {
        Self {
            num_searches: 5,
            max_pivots: 5,
            tol_margin: 10.0,
        }
    }
}

impl DefaultTreeGlobalPivotFinder {
    pub fn new(num_searches: usize, max_pivots: usize, tol_margin: f64) -> Self {
        Self {
            num_searches,
            max_pivots,
            tol_margin,
        }
    }

    /// Find global pivots by comparing `f` (true function) with `approx`
    /// (current approximation) at random + locally-optimized points.
    ///
    /// # Arguments
    /// - `local_dims`: dimension at each site
    /// - `f`: true function, f(multi_index) -> scalar magnitude
    /// - `approx`: current approximation, approx(multi_index) -> scalar magnitude
    /// - `abs_tol`: absolute tolerance (combined with tol_margin)
    /// - `rng`: random number generator
    ///
    /// The error at a point is `|f(x) - approx(x)|`.
    /// Points with error > `abs_tol * tol_margin` are candidates.
    pub fn find_pivots<F, G>(
        &self,
        local_dims: &[usize],
        f: &F,
        approx: &G,
        abs_tol: f64,
        rng: &mut impl Rng,
    ) -> Vec<MultiIndex>
    where
        F: Fn(&[usize]) -> f64,
        G: Fn(&[usize]) -> f64,
    {
        let n = local_dims.len();
        let threshold = abs_tol * self.tol_margin;
        let mut found: Vec<(MultiIndex, f64)> = Vec::new();

        for _ in 0..self.num_searches {
            // Random starting point
            let mut point: MultiIndex = (0..n)
                .map(|p| rng.random_range(0..local_dims[p]))
                .collect();

            // Greedy local search: sweep all dimensions
            let mut best_error = 0.0f64;
            let mut best_point = point.clone();

            for p in 0..n {
                for v in 0..local_dims[p] {
                    point[p] = v;
                    let error = (f(&point) - approx(&point)).abs();
                    if error > best_error {
                        best_error = error;
                        best_point = point.clone();
                    }
                }
                // Move to best value found for this dimension
                point = best_point.clone();
            }

            if best_error > threshold {
                found.push((best_point, best_error));
            }
        }

        // Sort by error descending, deduplicate, truncate
        found.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        found.dedup_by(|a, b| a.0 == b.0);
        found.truncate(self.max_pivots);
        found.into_iter().map(|(pivot, _)| pivot).collect()
    }
}
```

注意: TCI2 の finder は `Fn(&MultiIndex) -> T` (ジェネリック型)を取るが、TreeTCI 版は
`f` と `approx` を分離して受け取る設計。これにより:
- `f` は batch_eval から作った point eval closure
- `approx` は materialized TreeTN の evaluate を使った closure
として呼び出し側で組み立てる。型パラメータ T に依存しない (f64 の誤差計算のみ)。

complex の場合: 呼び出し側で `|idx| { let v = f_complex(idx); v.abs_sq().sqrt() }` のように
abs 値に変換してから渡す。

- [ ] **Step 3: Add module to lib.rs**

In `crates/tensor4all-treetci/src/lib.rs`, add:
```rust
pub mod globalpivot;
pub use globalpivot::DefaultTreeGlobalPivotFinder;
```

- [ ] **Step 4: Run test**

Run: `cargo nextest run --release -p tensor4all-treetci test_find_global_pivots`

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-treetci/src/globalpivot.rs crates/tensor4all-treetci/src/lib.rs
git commit -m "feat(treetci): add DefaultTreeGlobalPivotFinder"
```

---

## Task 3: optimize ループに global pivot search を組み込み

**Files:**
- Modify: `crates/tensor4all-treetci/src/optimize.rs`

- [ ] **Step 1: optimize_with_proposer に global search を追加**

`optimize_with_proposer` 関数の型境界に `FullPivLuScalar` を追加
（`to_treetn` に必要）。ループの `_iter` ごとに、`global_search_interval > 0` かつ
`iter % interval == 0` のとき global pivot search を実行する。

```rust
use crate::globalpivot::DefaultTreeGlobalPivotFinder;
use crate::materialize::{to_treetn, FullPivLuScalar};

pub fn optimize_with_proposer<T, F, P>(
    state: &mut SimpleTreeTci<T>,
    batch_eval: F,
    options: &TreeTciOptions,
    proposer: &P,
) -> Result<(Vec<usize>, Vec<f64>)>
where
    T: FullPivLuScalar,  // was: Scalar. Tightened for to_treetn
    DenseFaerLuKernel: PivotKernel<T>,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    // ... existing setup ...

    let global_finder = if options.global_search_interval > 0 {
        Some(DefaultTreeGlobalPivotFinder::new(
            options.num_global_searches,
            options.max_global_pivots,
            options.global_search_tol_margin,
        ))
    } else {
        None
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    for iter in 0..options.max_iter {
        // ... existing inner edge passes ...

        ranks.push(state.max_rank());
        let normalized_error = /* ... existing ... */;
        errors.push(normalized_error);

        // Global pivot search
        if let Some(ref finder) = global_finder {
            if (iter + 1) % options.global_search_interval == 0 {
                let error_scale = if options.normalize_error && state.max_sample_value > 0.0 {
                    state.max_sample_value
                } else {
                    1.0
                };
                let abs_tol = options.tolerance * error_scale;

                // Materialize current state
                let treetn = to_treetn(state, &batch_eval, Some(0))?;

                // Build point eval closures
                let point_eval_f = |idx: &[usize]| -> f64 {
                    let batch_data: Vec<usize> = idx.to_vec();
                    let batch = GlobalIndexBatch::new(&batch_data, idx.len(), 1).unwrap();
                    match batch_eval(batch) {
                        Ok(vals) => T::abs_val(vals[0]),
                        Err(_) => 0.0,
                    }
                };

                let point_eval_approx = |idx: &[usize]| -> f64 {
                    let n_sites = state.local_dims.len();
                    let index_values: std::collections::HashMap<usize, Vec<usize>> =
                        (0..n_sites).map(|s| (s, vec![idx[s]])).collect();
                    match treetn.evaluate(&index_values) {
                        Ok(scalar) => scalar.abs(),
                        Err(_) => 0.0,
                    }
                };

                let pivots = finder.find_pivots(
                    &state.local_dims,
                    &point_eval_f,
                    &point_eval_approx,
                    abs_tol,
                    &mut rng,
                );

                if !pivots.is_empty() {
                    state.add_global_pivots(&pivots)?;
                }
            }
        }

        // Early exit on convergence (existing)
    }

    Ok((ranks, errors))
}
```

**重要な変更点:**
- 型境界: `T: Scalar` → `T: FullPivLuScalar` (`to_treetn` が必要)
- `optimize_default` も同じ型境界に更新が必要
- `point_eval_f` で complex の場合は `T::abs_val` で f64 に変換
- `point_eval_approx` は `AnyScalar::abs()` で f64 に変換

`Scalar` → `FullPivLuScalar` の変更が既存コードに影響しないか確認:
`FullPivLuScalar` は `f32, f64, Complex32, Complex64` に実装済み。
`Scalar` のスーパートレイトなので、既存の呼び出し側が `f64` や `Complex64` を使っていれば問題なし。

- [ ] **Step 2: 早期終了条件に global pivot を考慮**

現在は `max_iter` まで常に回っている。収束判定を追加:

```rust
        // Early exit: error below tolerance AND no global pivots added
        if normalized_error < options.tolerance {
            // If global search is enabled, only stop if last search found nothing
            if global_finder.is_none() {
                break;
            }
            // Otherwise continue until next global search finds nothing
        }
```

（完全な収束判定は TCI2 の `convergence_criterion` を参考にするが、MVP ではシンプルに）

- [ ] **Step 3: Verify it compiles**

Run: `cargo build -p tensor4all-treetci --release`

- [ ] **Step 4: Commit**

```bash
git add crates/tensor4all-treetci/src/optimize.rs
git commit -m "feat(treetci): integrate global pivot search into optimize loop"
```

---

## Task 4: テスト — チェーン木 + nasty function (TCI.jl parity)

**Files:**
- Create or modify: `crates/tensor4all-treetci/tests/global_pivot.rs`

- [ ] **Step 1: Write integration test**

Create `crates/tensor4all-treetci/tests/global_pivot.rs`:

```rust
//! Integration test: global pivot search improves convergence on difficult functions.
//!
//! Adapted from TensorCrossInterpolation.jl test_tensorci2.jl "globalsearch" test.
//! Uses a chain tree (equivalent to MPS) with a nasty oscillatory function
//! that requires global pivot search for convergence.

use tensor4all_treetci::{
    crossinterpolate_tree_with_proposer, SimpleProposer,
    TreeTciEdge, TreeTciGraph, TreeTciOptions,
};
use std::f64::consts::PI;

/// Chain tree: 0--1--2--...--N-1
fn chain_graph(n: usize) -> TreeTciGraph {
    let edges: Vec<TreeTciEdge> = (0..n - 1)
        .map(|i| TreeTciEdge::new(i, i + 1))
        .collect();
    TreeTciGraph::new(n, &edges).unwrap()
}

/// Quantics-like encoding: bitlist → x in [0, 1)
fn bits_to_x(bits: &[usize], n_bits: usize) -> f64 {
    let mut x = 0.0;
    for (i, &b) in bits.iter().enumerate() {
        x += (b as f64) * 2.0f64.powi(-(i as i32 + 1));
    }
    x
}

/// Nasty oscillatory function from TCI.jl test suite:
/// f(x) = exp(-10x) * sin(2π * 100 * x^1.1)
fn nasty_function(x: f64) -> f64 {
    (-10.0 * x).exp() * (2.0 * PI * 100.0 * x.powf(1.1)).sin()
}

#[test]
fn global_pivot_search_improves_convergence_on_nasty_function() {
    let n_bits = 10;
    let graph = chain_graph(n_bits);
    let local_dims = vec![2; n_bits];

    let f = |idx: &[usize]| -> f64 {
        let x = bits_to_x(idx, n_bits);
        nasty_function(x)
    };

    // Without global pivot search
    let options_no_global = TreeTciOptions {
        tolerance: 1e-8,
        max_iter: 30,
        max_bond_dim: 100,
        normalize_error: true,
        global_search_interval: 0, // disabled
        ..Default::default()
    };

    let (_, _ranks_no, errors_no) = crossinterpolate_tree_with_proposer(
        f,
        None::<fn(tensor4all_treetci::GlobalIndexBatch<'_>) -> anyhow::Result<Vec<f64>>>,
        local_dims.clone(),
        graph.clone(),
        vec![vec![0; n_bits]],
        options_no_global,
        Some(0),
        &SimpleProposer::default(),
    ).unwrap();

    let error_no_global = errors_no.last().copied().unwrap_or(f64::INFINITY);

    // With global pivot search
    let options_global = TreeTciOptions {
        tolerance: 1e-8,
        max_iter: 30,
        max_bond_dim: 100,
        normalize_error: true,
        global_search_interval: 1, // every iteration
        num_global_searches: 10,
        max_global_pivots: 5,
        global_search_tol_margin: 10.0,
    };

    let (_, _ranks_yes, errors_yes) = crossinterpolate_tree_with_proposer(
        f,
        None::<fn(tensor4all_treetci::GlobalIndexBatch<'_>) -> anyhow::Result<Vec<f64>>>,
        local_dims.clone(),
        graph.clone(),
        vec![vec![0; n_bits]],
        options_global,
        Some(0),
        &SimpleProposer::default(),
    ).unwrap();

    let error_global = errors_yes.last().copied().unwrap_or(f64::INFINITY);

    // Global pivot search should achieve better or equal convergence
    // The nasty function typically requires global pivots for good convergence
    eprintln!(
        "Without global: error={:.2e}, With global: error={:.2e}",
        error_no_global, error_global
    );

    // At minimum, with global search should converge below tolerance
    assert!(
        error_global < 1e-6,
        "Global pivot search should help converge: got {:.2e}",
        error_global
    );
}
```

注意: `crossinterpolate_tree_with_proposer` のシグネチャが `FullPivLuScalar` に変わるため、
`f64` は問題なし。`point_eval` と `batch_eval` の両方を渡す現在の API に合わせる。

テスト関数は TCI.jl の `exp(-10x) * sin(2π * 100 * x^1.1)` そのもの。
quantics grid の代わりに手動の `bits_to_x` で binary → float 変換。

- [ ] **Step 2: Run test**

Run: `cargo nextest run --release -p tensor4all-treetci global_pivot`

Expected: PASS. Global pivot search version achieves error < 1e-6.

- [ ] **Step 3: Run full test suite**

Run: `cargo nextest run --release --workspace`

Expected: No regressions. `FullPivLuScalar` 型境界の変更が既存テストに影響しないことを確認。

- [ ] **Step 4: fmt + clippy**

Run: `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings`

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-treetci/tests/global_pivot.rs \
       crates/tensor4all-treetci/src/globalpivot.rs \
       crates/tensor4all-treetci/src/optimize.rs \
       crates/tensor4all-treetci/src/lib.rs
git commit -m "feat(treetci): global pivot search with nasty function test"
```

---

## 設計上の注意点

1. **`Scalar` → `FullPivLuScalar` 型境界の変更**
   - `optimize_default` と `optimize_with_proposer` の両方で変更
   - `FullPivLuScalar: Scalar + TensorElement` なのでスーパートレイト
   - `f32, f64, Complex32, Complex64` に実装済み → 既存コードに影響なし
   - ただし `crossinterpolate_tree` / `crossinterpolate_tree_with_proposer` の型境界も連鎖的に更新が必要

2. **Materialization コスト**
   - `to_treetn` は毎回全テンソルを再構築 → `global_search_interval` で頻度制御
   - デフォルト 0 (無効) なので、パフォーマンスへの影響はユーザーが opt-in

3. **Complex 対応**
   - `find_pivots` は `Fn(&[usize]) -> f64` を受け取る（実数の誤差値）
   - 呼び出し側で `T::abs_val()` / `AnyScalar::abs()` で f64 に変換
   - Complex64 でも同じコードパスで動作

4. **RNG**
   - `StdRng::seed_from_u64(42)` で決定論的。テストの再現性を保証
   - 将来的に `TreeTciOptions` に seed パラメータを追加可能
