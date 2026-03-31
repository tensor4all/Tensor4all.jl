# TreeTCI Global Pivot Search 詳細設計

## Overview

TreeTCI の optimize ループに global pivot search を追加し、局所的な特徴を持つ関数の収束を改善する。

## スコープ

1. `TreeTN::evaluate_batch` — batch 版 evaluate を TreeTN に追加
2. `DefaultTreeGlobalPivotFinder` — greedy local search (TCI2 の finder と同アルゴリズム)
3. `TreeTciOptions` にパラメータ追加
4. `optimize_with_proposer` ループに組み込み
5. テスト: chain tree + TCI.jl parity の nasty function

## スコープ外

- `TreeTN::evaluate_batch` の部分木キャッシュ最適化 (将来タスク)
- index type の generic 化 (usize 固定のまま)
- C API / Julia 側の変更 (Rust のみ)

---

## 1. TreeTN::evaluate — batch 版に置き換え

### 設計方針

batch evaluation が唯一の API。`evaluate` という名前で batch を受け取る。
1点評価は `evaluate(&[single_idx])` で表現。

### 現状

```rust
// crates/tensor4all-treetn/src/treetn/ops.rs
impl<T: TensorLike, V: ...> TreeTN<T, V> {
    pub fn evaluate(
        &self,
        index_values: &HashMap<V, Vec<usize>>,
    ) -> Result<AnyScalar>
}
```

1点ずつ `HashMap` を構築して評価。

### 変更

既存の 1点版 `evaluate` を **削除** し、batch 版 `evaluate` に置き換える。
1点評価は `evaluate(&[single_idx])` で表現。

```rust
// crates/tensor4all-treetn/src/treetn/ops.rs

/// Evaluate the TreeTN at multiple multi-indices (batch).
///
/// Each element of `indices` is a HashMap mapping vertex names to their
/// site index values. Returns one AnyScalar per evaluation point.
///
/// For single-point evaluation, pass a slice of length 1.
///
/// Internal implementation evaluates point-by-point.
/// Future: subtree cache optimization for shared prefixes.
pub fn evaluate(
    &self,
    indices: &[HashMap<V, Vec<usize>>],
) -> Result<Vec<AnyScalar>>
where
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    // existing single-point logic inlined here, called per point
    indices.iter().map(|idx| {
        // ... existing evaluate body (onehot contraction) ...
    }).collect()
}
```

旧 `evaluate(&HashMap)` の呼び出し箇所を全て `evaluate(&[idx])` に更新する。
これは **破壊的変更** だが、early development のため後方互換性不要 (AGENTS.md)。

### TreeTCI 向けヘルパー

vertex 名 = `usize` で各 vertex が 1 site index のみ持つケースに特化:

```rust
/// Evaluate at multiple multi-indices given as flat site-order arrays.
///
/// 各 multi-index は `[idx_site0, idx_site1, ..., idx_site_{n-1}]` (0-based)。
/// vertex 名が usize の TreeTN 専用。
///
/// 1点評価: `evaluate_at_site_indices(&[vec![0, 1, 2]])`
pub fn evaluate_at_site_indices(
    &self,
    indices: &[Vec<usize>],
) -> Result<Vec<AnyScalar>>
where
    V: From<usize> + Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let n_sites = self.node_count();
    let hash_indices: Vec<HashMap<V, Vec<usize>>> = indices.iter()
        .map(|multi_idx| {
            (0..n_sites)
                .map(|s| (V::from(s), vec![multi_idx[s]]))
                .collect()
        })
        .collect();
    self.evaluate(&hash_indices)
}
```

### C API への影響

既存の `t4a_treetn_evaluate_batch` は内部で旧 1点版 `evaluate` を
point ごとに呼んでいる。新しい batch 版 `evaluate` を使うように更新。

### 既存コードの呼び出し箇所の更新

旧 `evaluate(&HashMap)` を呼んでいる箇所を全て
`evaluate(&[hashmap])[0]` に置き換える。grep で洗い出して全更新。

---

## 2. TreeTciOptions パラメータ追加

```rust
// crates/tensor4all-treetci/src/optimize.rs

#[derive(Clone, Debug)]
pub struct TreeTciOptions {
    // 既存
    pub tolerance: f64,
    pub max_iter: usize,
    pub max_bond_dim: usize,
    pub normalize_error: bool,

    // Global pivot search
    /// Run global pivot search every N iterations. 0 = disabled (default).
    pub global_search_interval: usize,
    /// Maximum number of global pivots to add per search round.
    pub max_global_pivots: usize,
    /// Number of random starting points for greedy search.
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
            global_search_interval: 0,
            max_global_pivots: 5,
            num_global_searches: 5,
            global_search_tol_margin: 10.0,
        }
    }
}
```

---

## 3. DefaultTreeGlobalPivotFinder

### ファイル

`crates/tensor4all-treetci/src/globalpivot.rs`

### アルゴリズム

TCI2 の `DefaultGlobalPivotFinder` と同じ greedy local search。
全ての関数評価は **batch** で行う。

```
for each of num_searches random starting points:
    current_point = random_point(local_dims)
    best_error = 0
    best_point = current_point

    for each site p in 0..n_sites:
        # Batch: local_dims[p] 個の候補点を一括評価
        candidates = [current_point with site p = v for v in 0..local_dims[p]]

        f_vals = batch_eval(candidates)           # 真値 (batch)
        approx_vals = treetn.evaluate_at_site_indices(candidates)  # 近似値

        errors = |f_vals - approx_vals|
        v_best = argmax(errors)

        if errors[v_best] > best_error:
            best_error = errors[v_best]
            best_point = candidates[v_best]

        current_point[p] = v_best  # 次の dimension は最良値から出発

    if best_error > abs_tol * tol_margin:
        found_pivots.append(best_point)

sort by error descending, dedup, truncate to max_pivots
```

### 型シグネチャ

```rust
use crate::{GlobalIndexBatch, OwnedGlobalIndexBatch};
use crate::materialize::FullPivLuScalar;
use tensor4all_tcicore::MultiIndex;
use tensor4all_treetn::TreeTN;
use tensor4all_core::TensorDynLen;
use rand::Rng;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct DefaultTreeGlobalPivotFinder {
    pub num_searches: usize,
    pub max_pivots: usize,
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
        Self { num_searches, max_pivots, tol_margin }
    }

    /// Find global pivots where interpolation error is large.
    ///
    /// # Arguments
    /// - `local_dims`: site dimensions
    /// - `batch_eval`: the user's batch evaluation function (true values)
    /// - `treetn`: materialized current approximation
    /// - `abs_tol`: absolute tolerance threshold
    /// - `rng`: random number generator
    ///
    /// # Returns
    /// Multi-indices where |f(x) - approx(x)| > abs_tol * tol_margin
    pub fn find_pivots<T, F>(
        &self,
        local_dims: &[usize],
        batch_eval: &F,
        treetn: &TreeTN<TensorDynLen, usize>,
        abs_tol: f64,
        rng: &mut impl Rng,
    ) -> Result<Vec<MultiIndex>>
    where
        T: FullPivLuScalar,
        F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    {
        // ... implementation ...
    }
}
```

### Batch 評価の実装詳細

dimension sweep の各 site `p` で `local_dims[p]` 個の候補点を作り、
一括で `batch_eval` と `treetn.evaluate_at_site_indices` を呼ぶ。

```rust
// site p の sweep: local_dims[p] 個の候補
let d = local_dims[p];
let mut candidates: Vec<Vec<usize>> = Vec::with_capacity(d);
for v in 0..d {
    let mut point = current_point.clone();
    point[p] = v;
    candidates.push(point);
}

// batch_eval で真値を取得
let batch_data: Vec<usize> = candidates.iter()
    .flat_map(|c| c.iter().copied())
    .collect();
let batch = GlobalIndexBatch::new(&batch_data, n_sites, d)?;
let f_vals: Vec<T> = batch_eval(batch)?;

// treetn で近似値を取得
let approx_vals: Vec<AnyScalar> = treetn.evaluate_at_site_indices(&candidates)?;

// 誤差計算
for i in 0..d {
    let f_abs = T::abs_val(f_vals[i]);
    let approx_abs = approx_vals[i].abs();
    let error = (f_abs - approx_abs).abs();
    // ... track best ...
}
```

注意: `f_vals` は `T` 型 (f64 or Complex64), `approx_vals` は `AnyScalar`。
誤差は `|f(x)| - |approx(x)|` ではなく `|f(x) - approx(x)|` であるべき。

**Complex 対応の誤差計算:**

```rust
// T から f64 への abs 変換
let f_abs = f64::sqrt(T::abs_sq(f_vals[i]));
// AnyScalar からの abs
let approx_abs_re = approx_vals[i].real();
let approx_abs_im = approx_vals[i].imag();

// 差の abs: |f - approx|
// f_vals[i] を re/im に分解する必要がある
// T: FullPivLuScalar は Scalar trait を持つので:
let f_re: f64 = /* T の実部取得 */;
let f_im: f64 = /* T の虚部取得 */;
let diff_re = f_re - approx_abs_re;
let diff_im = f_im - approx_abs_im;
let error = (diff_re * diff_re + diff_im * diff_im).sqrt();
```

`Scalar` trait に `real_part()`, `imag_part()` -> f64 があるか要確認。
なければ f64 の場合は `(val as f64, 0.0)`, Complex64 は `(val.re, val.im)` で
マッチする。

実装上は `T` の concrete type (`f64` or `Complex64`) でディスパッチする
ヘルパー関数を使う:

```rust
fn scalar_to_re_im<T: Scalar>(val: T) -> (f64, f64) {
    // T::abs_sq は f64 を返す (Scalar trait)
    // T が f64 なら (val, 0.0)
    // T が Complex64 なら (val.re, val.im)
    // Scalar trait に直接 re/im アクセスがなければ:
    // val を AnyScalar に変換して .real(), .imag() を使うのが安全
    todo!("check Scalar trait for re/im access")
}
```

→ 実装時に `Scalar` trait のメソッドを確認して適切な変換を選ぶ。

---

## 4. optimize ループへの組み込み

### 型境界の変更

```rust
pub fn optimize_with_proposer<T, F, P>(
    state: &mut SimpleTreeTci<T>,
    batch_eval: F,
    options: &TreeTciOptions,
    proposer: &P,
) -> Result<(Vec<usize>, Vec<f64>)>
where
    T: FullPivLuScalar,  // was: Scalar (tightened for to_treetn)
    DenseFaerLuKernel: PivotKernel<T>,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
```

`FullPivLuScalar: Scalar + TensorElement` なので上位互換。
`f32, f64, Complex32, Complex64` に実装済み。

### ループ変更

```rust
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
let mut nglobal_pivots_history: Vec<usize> = Vec::new();

for iter in 0..options.max_iter {
    // --- 既存の inner edge passes ---
    for _pass in 0..INNER_EDGE_PASSES {
        // ... update_edge ...
    }

    ranks.push(state.max_rank());
    let normalized_error = /* 既存 */;
    errors.push(normalized_error);

    // --- Global pivot search ---
    let n_global = if let Some(ref finder) = global_finder {
        if (iter + 1) % options.global_search_interval == 0 {
            let error_scale = if options.normalize_error && state.max_sample_value > 0.0 {
                state.max_sample_value
            } else {
                1.0
            };
            let abs_tol = options.tolerance * error_scale;

            // Materialize current state
            let treetn = to_treetn(state, &batch_eval, Some(0))?;

            // Find global pivots
            let pivots = finder.find_pivots::<T, _>(
                &state.local_dims,
                &batch_eval,
                &treetn,
                abs_tol,
                &mut rng,
            )?;

            let n = pivots.len();
            if !pivots.is_empty() {
                state.add_global_pivots(&pivots)?;
            }
            n
        } else {
            0
        }
    } else {
        0
    };
    nglobal_pivots_history.push(n_global);

    // --- Early exit ---
    if normalized_error < options.tolerance {
        // global search が有効なら、直近の search で 0 pivots のとき終了
        if global_finder.is_none() || n_global == 0 {
            break;
        }
    }
}
```

### 戻り値の変更

現在は `(Vec<usize>, Vec<f64>)` (ranks, errors)。
Global pivot 情報も返すとデバッグに有用:

```rust
/// Optimization result.
pub struct OptimizeResult {
    pub ranks: Vec<usize>,
    pub errors: Vec<f64>,
    pub nglobal_pivots: Vec<usize>,
}
```

→ **破壊的変更**になるため、今回は `nglobal_pivots` は内部で使うだけにして
戻り値は変えない。将来 `OptimizeResult` 構造体に移行。

---

## 5. テスト

### テスト関数: TCI.jl parity

```rust
// crates/tensor4all-treetci/tests/global_pivot.rs

/// Chain tree: 0--1--2--...--N-1
fn chain_graph(n: usize) -> TreeTciGraph { ... }

/// Quantics-like: bitlist [b0, b1, ...] → x = sum(b_i * 2^{-(i+1)}) ∈ [0, 1)
fn bits_to_x(bits: &[usize]) -> f64 { ... }

/// TCI.jl "nasty function": f(x) = exp(-10x) * sin(2π * 100 * x^1.1)
fn nasty_function(x: f64) -> f64 {
    (-10.0 * x).exp() * (2.0 * PI * 100.0 * x.powf(1.1)).sin()
}
```

### テストケース

**1. global search あり vs なし の比較:**
- 10-bit chain (1024 grid points)
- `global_search_interval = 1` (毎イテレーション) vs `0` (無効)
- global search あり → error < 1e-6 を期待
- global search なし → error がより大きい（または収束しない）を期待

**2. global search の interval 動作確認:**
- `global_search_interval = 3` で、3イテレーションごとに search が走ることを確認
  (nglobal_pivots_history をチェック)

**3. 既存テストのリグレッション:**
- `global_search_interval = 0` (デフォルト) で既存の動作が変わらないことを確認

---

## 実装順序

| Task | 内容 | ファイル |
|------|------|---------|
| 1 | `TreeTN::evaluate_at_site_indices` | `crates/tensor4all-treetn/src/treetn/ops.rs` |
| 2 | `TreeTciOptions` パラメータ追加 | `crates/tensor4all-treetci/src/optimize.rs` |
| 3 | `DefaultTreeGlobalPivotFinder` | `crates/tensor4all-treetci/src/globalpivot.rs` + `lib.rs` |
| 4 | `optimize_with_proposer` 組み込み | `crates/tensor4all-treetci/src/optimize.rs` |
| 5 | テスト: nasty function | `crates/tensor4all-treetci/tests/global_pivot.rs` |
| 6 | 全テスト + clippy + fmt | validation |

---

## 設計上の注意

1. **型境界 `Scalar` → `FullPivLuScalar`**: optimize の型境界を狭める。
   `FullPivLuScalar` は `f32, f64, Complex32, Complex64` に実装済みなので
   既存コードに影響なし。

2. **Materialization コスト**: `to_treetn` は全テンソルを再構築。
   `global_search_interval` で頻度制御。デフォルト 0 (無効) なので opt-in。

3. **Complex 対応**: `find_pivots` 内の誤差計算で `T` → `(re, im)` 変換が必要。
   `AnyScalar` は `.real()`, `.imag()` を持つ。`T` 側は Scalar trait の
   メソッドで対応（実装時に確認）。

4. **RNG**: `StdRng::seed_from_u64(42)` で決定論的。

5. **戻り値**: 今回は `(Vec<usize>, Vec<f64>)` のまま変えない。
