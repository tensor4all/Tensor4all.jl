# API 統一 & tensorci 削除 設計 (最終版)

## Overview

1. `tensor4all-core` に軽量 N 次元 column-major array 型を追加
2. `TreeTN::evaluate` を IndexId ベース batch API に移行（HashSet 順序バグ修正）
3. TreeTCI API を統一（`crossinterpolate2`、batch evaluate のみ）
4. tensorci (chain TCI) の C API と Julia ラッパーを削除
5. TreeTCI 内部状態を `ColMajorArray<usize>` に移行

## 動機

- HashMap ベースの evaluate は遅い + HashSet 順序バグ
- `Vec<Vec<usize>>` (nested array) はキャッシュ非効率
- `point_eval` と `batch_eval` の二重 API は不要
- tensorci (chain) は treetci の下位互換 → 削除して保守コスト削減
- C API 命名が不一致 → `t4a_<TYPE>_<OP>` を徹底

---

## Part 1: 削除

### tensor4all-capi: tensorci C API 削除

| 削除対象 | ファイル |
|---------|---------|
| `tensorci.rs` 全体 | `crates/tensor4all-capi/src/tensorci.rs` |
| テスト | `crates/tensor4all-capi/src/tensorci/tests/mod.rs` |
| types.rs から `t4a_tci2_f64`, `t4a_tci2_c64` | `crates/tensor4all-capi/src/types.rs` |
| lib.rs から `mod tensorci; pub use tensorci::*;` | `crates/tensor4all-capi/src/lib.rs` |

**削除される C API 関数:**
- `t4a_tci2_f64_*` (18 関数)
- `t4a_tci2_c64_*` (18 関数)
- `t4a_crossinterpolate2_f64`, `t4a_crossinterpolate2_c64`
- `t4a_estimate_true_error_f64`, `t4a_opt_first_pivot_f64`
- `EvalCallback`, `EvalCallbackC64` 型

### Tensor4all.jl: TensorCI Julia ラッパー削除

| 削除対象 | ファイル |
|---------|---------|
| TensorCI モジュール | `src/TensorCI.jl` |
| テスト | `test/test_tensorci.jl`, `test/test_tensorci_advanced.jl` |
| Tensor4all.jl から `include("TensorCI.jl")` | `src/Tensor4all.jl` |
| runtests.jl から include | `test/runtests.jl` |

### Tensor4all.jl: SimpleTT Julia ラッパーの tensorci 依存除去

`SimpleTT.jl` が `TensorCI.jl` に依存している箇所があれば除去。
`to_tensor_train` 等の TCI → SimpleTT 変換は TreeTCI 側で提供。

---

## Part 2: ColMajorArray 型 (`tensor4all-core`)

### ファイル

`crates/tensor4all-core/src/col_major_array.rs`

### 3つの型

```rust
/// Borrowed N-dimensional column-major array view.
/// Element at [i0, i1, ...] is at: i0 + shape[0] * (i1 + shape[1] * ...)
#[derive(Clone, Copy, Debug)]
pub struct ColMajorArrayRef<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
}

/// Mutable borrowed N-dimensional column-major array view.
#[derive(Debug)]
pub struct ColMajorArrayMut<'a, T> {
    data: &'a mut [T],
    shape: &'a [usize],
}

/// Owned N-dimensional column-major array.
#[derive(Clone, Debug)]
pub struct ColMajorArray<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}
```

`T` に trait 境界なし。

### API

**全型共通:**
```rust
fn ndim(&self) -> usize
fn shape(&self) -> &[usize]
fn len(&self) -> usize              // 全要素数
fn is_empty(&self) -> bool
fn data(&self) -> &[T]
fn get(&self, indices: &[usize]) -> Option<&T>
```

**Mut, Owned:**
```rust
fn data_mut(&mut self) -> &mut [T]
fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T>
```

**Owned のみ:**
```rust
fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self>
fn into_data(self) -> Vec<T>
fn as_ref(&self) -> ColMajorArrayRef<'_, T>
fn as_mut(&mut self) -> ColMajorArrayMut<'_, T>
```

**2D 便利メソッド (Owned のみ):**
```rust
fn nrows(&self) -> usize            // shape[0], panics if not 2D
fn ncols(&self) -> usize            // shape[1], panics if not 2D
fn column(&self, j: usize) -> Option<&[T]>
fn push_column(&mut self, column: &[T]) -> Result<()>  // column-major append
```

**ファクトリ:**
```rust
fn filled(shape: Vec<usize>, value: T) -> Self  // T: Clone
fn zeros(shape: Vec<usize>) -> Self              // T: Default + Clone
```

### flat_offset (checked arithmetic)

```rust
fn flat_offset(shape: &[usize], indices: &[usize]) -> Option<usize> {
    if indices.len() != shape.len() { return None; }
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&idx, &dim) in indices.iter().zip(shape.iter()) {
        if idx >= dim { return None; }
        offset = offset.checked_add(idx.checked_mul(stride)?)?;
        stride = stride.checked_mul(dim)?;
    }
    Some(offset)
}
```

---

## Part 3: TreeTN API 変更

### all_site_index_ids (新規)

```rust
/// Returns all site index IDs and their owning vertex names.
///
/// Returns (index_ids, vertex_names) where index_ids[i] belongs to
/// vertex vertex_names[i]. Order is unspecified but consistent
/// between the two vectors.
///
/// For evaluate(), pass index_ids and arrange values in the same order.
pub fn all_site_index_ids(&self) -> Result<(
    Vec<<T::Index as IndexLike>::Id>,
    Vec<V>,
)>
where
    V: Clone,
    <T::Index as IndexLike>::Id: Clone,
```

### evaluate (新規、旧版を置き換え)

```rust
/// Evaluate the TreeTN at multiple multi-indices (batch).
///
/// index_ids: which indices to fix (n_indices 個).
///   Each ID identifies a specific site index unambiguously.
///   Must enumerate every site index exactly once.
/// values: shape = [n_indices, n_points], column-major.
///   values.get(&[i, p]) = value of index_ids[i] at point p.
///
/// Returns one AnyScalar per point.
pub fn evaluate(
    &self,
    index_ids: &[<T::Index as IndexLike>::Id],
    values: ColMajorArrayRef<'_, usize>,
) -> Result<Vec<AnyScalar>>
where
    <T::Index as IndexLike>::Id:
        Clone + Hash + Eq + Ord + Debug + Send + Sync,
```

**内部実装:**
- `index_ids` から各 index を ID で直接 lookup（HashMap 不要）
- point ごとに onehot contraction（既存ロジック流用）
- HashSet 順序バグ解消 — index を ID で直接指定するため

**旧 `evaluate(&HashMap<V, Vec<usize>>)` は削除。**

### C API

**削除:** `t4a_treetn_evaluate_batch` (旧 vertex 名ベース)

**新規:**

```rust
/// Get all site index IDs and vertex names.
/// Query-then-fill: pass NULL buffers to get out_n_indices only.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_all_site_index_ids(
    ptr: *const t4a_treetn,
    out_index_ids: *mut u64,           // DynId as u64, NULL for query
    out_vertex_names: *mut libc::size_t, // vertex name (usize), NULL for query
    buf_len: libc::size_t,
    out_n_indices: *mut libc::size_t,
) -> StatusCode;

/// Evaluate TreeTN at multiple points.
/// index_ids: n_indices index IDs (from all_site_index_ids)
/// values: column-major [n_indices, n_points]
/// out_re: n_points results (real part)
/// out_im: n_points results (imag part, NULL for real-only)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetn_evaluate(
    ptr: *const t4a_treetn,
    index_ids: *const u64,
    n_indices: libc::size_t,
    values: *const libc::size_t,
    n_points: libc::size_t,
    out_re: *mut libc::c_double,
    out_im: *mut libc::c_double,
) -> StatusCode;
```

### 呼び出し箇所の更新

| ファイル | 変更 |
|---------|------|
| `treetn/tests/ops.rs` (6箇所) | `HashMap::from(...)` → `all_site_index_ids` + `ColMajorArrayRef` |
| `treetci/tests/simple_parity.rs` (4箇所) | 同上 |
| `treetci/tests/advanced_quantics.rs` (1箇所) | 同上 |
| `treetci/src/materialize/tests.rs` | 同上 |
| `tensor4all-capi/src/treetn.rs` | 旧 `evaluate_batch` → 新 `evaluate` |

---

## Part 4: TreeTCI API 統一

### 命名変更

| Before | After |
|--------|-------|
| `crossinterpolate_tree` | `crossinterpolate2` |
| `crossinterpolate_tree_with_proposer` | 削除（`crossinterpolate2` に統合） |
| `batch_eval` (引数名) | `evaluate` |
| `point_eval` | 削除 |
| `fallback_batch_eval` | 削除 |
| `TreeTciBatchEvalCallback` (C API) | `TreeTciEvalCallback` |
| `TreeTciBatchEvalCallbackC64` (C API) | `TreeTciEvalCallbackC64` |
| `t4a_crossinterpolate_tree_f64` (C API) | `t4a_treetci_crossinterpolate2_f64` |
| `t4a_crossinterpolate_tree_c64` (C API) | `t4a_treetci_crossinterpolate2_c64` |

### crossinterpolate2 (統一版)

```rust
pub fn crossinterpolate2<T, F, P>(
    evaluate: F,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: TreeTciOptions,
    center_site: Option<usize>,
    proposer: &P,
) -> Result<TreeTciRunResult>
where
    T: FullPivLuScalar,
    DenseFaerLuKernel: PivotKernel<T>,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    let pivots = if initial_pivots.is_empty() {
        vec![vec![0; local_dims.len()]]
    } else {
        initial_pivots
    };

    let mut tci = SimpleTreeTci::<T>::new(local_dims, graph)?;
    tci.add_global_pivots(&pivots)?;

    // Initialize max_sample_value via batch evaluate
    let n_sites = tci.local_dims.len();
    let flat: Vec<usize> = pivots.iter().flat_map(|p| p.iter().copied()).collect();
    let shape = [n_sites, pivots.len()];
    let batch = GlobalIndexBatch::new(&flat, &shape)?;
    let init_vals = evaluate(batch)?;
    tci.max_sample_value = init_vals.iter()
        .map(|v| T::abs_val(*v))
        .fold(0.0f64, f64::max);
    ensure!(tci.max_sample_value > 0.0, ...);

    let (ranks, errors) = optimize_with_proposer(&mut tci, &evaluate, &options, proposer)?;
    let treetn = to_treetn(&tci, &evaluate, center_site)?;

    Ok((treetn, ranks, errors))
}
```

### GlobalIndexBatch

**shape 所有問題への対応:**

`GlobalIndexBatch` は pure newtype にせず、独自に shape 情報を所有：

```rust
#[derive(Clone, Copy, Debug)]
pub struct GlobalIndexBatch<'a> {
    data: &'a [usize],
    n_sites: usize,
    n_points: usize,
}

impl<'a> GlobalIndexBatch<'a> {
    pub fn new(data: &'a [usize], n_sites: usize, n_points: usize) -> Result<Self>;

    pub fn n_sites(&self) -> usize;
    pub fn n_points(&self) -> usize;
    pub fn data(&self) -> &'a [usize];
    pub fn get(&self, site: usize, point: usize) -> Option<usize>;

    /// Convert to ColMajorArrayRef (caller must provide shape with sufficient lifetime).
    pub fn as_col_major<'s>(&self, shape: &'s [usize]) -> Result<ColMajorArrayRef<'s, usize>>
    where 'a: 's;
}
```

**現在の構造を維持**。`ColMajorArrayRef` への変換メソッドを追加するだけ。

### optimize_with_proposer

```rust
pub fn optimize_with_proposer<T, F, P>(
    state: &mut SimpleTreeTci<T>,
    evaluate: &F,                     // renamed from batch_eval
    options: &TreeTciOptions,
    proposer: &P,
) -> Result<(Vec<usize>, Vec<f64>)>
where
    T: FullPivLuScalar,               // tightened from Scalar
    DenseFaerLuKernel: PivotKernel<T>,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
```

### to_treetn

```rust
pub fn to_treetn<T, F>(
    state: &SimpleTreeTci<T>,
    evaluate: &F,                     // renamed from batch_eval
    center_site: Option<usize>,
) -> Result<TreeTN<TensorDynLen, usize>>
where
    T: FullPivLuScalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
```

---

## Part 5: TreeTCI 内部状態

### ijset + ijset_history

```rust
// Before
pub ijset: HashMap<SubtreeKey, Vec<MultiIndex>>,
pub ijset_history: Vec<HashMap<SubtreeKey, Vec<MultiIndex>>>,

// After
pub ijset: HashMap<SubtreeKey, ColMajorArray<usize>>,
// shape = [n_subtree_sites, n_pivots]
// .ncols() = ピボット数
// .column(j) = j 番目のピボット
// .push_column(pivot) = ピボット追加

pub ijset_history: Vec<HashMap<SubtreeKey, ColMajorArray<usize>>>,
```

**空の初期状態:** `ColMajorArray::new(vec![], vec![n_subtree_sites, 0])` → 0列の 2D 配列。

**push_unique_column:**
```rust
fn push_unique_column(array: &mut ColMajorArray<usize>, column: &[usize]) {
    // 既存列と比較、重複なければ追加
    let nrows = array.nrows();
    for j in 0..array.ncols() {
        if array.column(j) == Some(column) {
            return; // duplicate
        }
    }
    array.push_column(column).unwrap();
}
```

### .len() → .ncols() への置き換え

全箇所でピボット数を参照する `.len()` を `.ncols()` に変更:
- `state.rs` L106
- `materialize.rs` L89, L217
- `proposer.rs` L82, L248

---

## Part 6: Julia ラッパー更新

### 削除

- `src/TensorCI.jl` 全体
- `test/test_tensorci.jl`, `test/test_tensorci_advanced.jl`
- `src/Tensor4all.jl` の TensorCI include と using
- `test/runtests.jl` の TensorCI include

### TreeTCI 更新

- `crossinterpolate_tree` → `crossinterpolate2` (Julia 側は既に改名済み、内部実装を更新)
- `evaluate` 関数: `all_site_index_ids` + 新 `t4a_treetn_evaluate` を使用
- C API シンボル名の更新 (`_sym` 呼び出し)

---

## 実装順序

| Task | 内容 | crate |
|------|------|-------|
| 1 | `ColMajorArray` 型を core に追加 + テスト | tensor4all-core |
| 2 | tensorci C API 削除 | tensor4all-capi |
| 3 | `TreeTN::all_site_index_ids` 追加 | tensor4all-treetn |
| 4 | `TreeTN::evaluate` を新 API に変更 + 旧削除 | tensor4all-treetn |
| 5 | 旧 evaluate 呼び出し箇所を全更新 | tests, treetci tests |
| 6 | TreeTCI: `evaluate` 改名 + `crossinterpolate2` 統一 + point_eval 削除 | tensor4all-treetci |
| 7 | TreeTCI: `ijset` + `ijset_history` → `ColMajorArray<usize>` | tensor4all-treetci |
| 8 | C API: treetci 改名 + treetn 新 API | tensor4all-capi |
| 9 | Julia: TensorCI 削除 + TreeTCI/TreeTN 更新 | Tensor4all.jl |
| 10 | 全テスト + clippy + fmt | validation |

---

## 設計上の注意

1. **`ColMajorArrayRef` の shape**: borrowed `&'a [usize]`。SmallVec 不使用。

2. **`GlobalIndexBatch`**: pure newtype にしない。独自に `n_sites, n_points` を所有する
   現在の設計を維持。`as_col_major()` 変換メソッドを追加。

3. **`TreeTN::evaluate` の IndexId**: `&[<T::Index as IndexLike>::Id]` で index を直接指定。
   `all_site_index_ids()` で事前取得。HashSet 順序バグ解消。
   `index_ids` は全 site index を exactly once 列挙する契約 (docstring に明記)。

4. **型境界 `FullPivLuScalar`**: `f32, f64, Complex32, Complex64` に実装済み。

5. **後方互換性不要** (AGENTS.md): 旧 API は即削除。

6. **ijset_history**: `ijset` と同時に `ColMajorArray` に移行。
   空配列は `[n_subtree_sites, 0]` shaped。

7. **DynId**: `pub struct DynId(pub u64)` → C API で `uint64_t` として渡せる。

8. **Python ラッパー**: 無視 (disable 方針)。

9. **tensorci Rust crate 自体は残す**: C API と Julia ラッパーのみ削除。
   Rust 内部で他 crate が依存している可能性があるため。
