# TreeTCI C API & Julia Wrapper Design

## Overview

tensor4all-rs の `tensor4all-treetci` クレートで実装された TreeTCI (tree-structured tensor cross interpolation) を C API 経由で Julia から利用できるようにする。

**スコープ:**
1. tensor4all-rs 側: `tensor4all-capi` に TreeTCI の C API バインディングを追加
2. Tensor4all.jl 側: C API を呼び出す Julia ラッパーモジュール `TreeTCI` を追加

**設計方針:**
- **ステートフル API (Approach B)**: `SimpleTreeTci` をオペークハンドルとして公開し、ピボット追加 → sweep → 検査 → materialization のライフサイクルを制御可能にする
- **単一バッチコールバック**: point eval と batch eval を統一。n_points=1 が point eval に相当
- **Proposer 列挙型選択**: `DefaultProposer`, `SimpleProposer`, `TruncatedDefaultProposer` を enum で切り替え
- **高レベル便利関数**: `crossinterpolate_tree` 一発実行版も提供

---

## Part 1: C API (tensor4all-rs 側)

### 1.1 ファイル構成

```
crates/tensor4all-capi/
  src/
    treetci.rs    # 新規: TreeTCI C API 関数すべて
    types.rs      # 既存: t4a_treetci_graph, t4a_treetci_f64 の型定義を追加
    lib.rs        # 既存: mod treetci; pub use treetci::*; を追加
  Cargo.toml      # 既存: tensor4all-treetci への依存を追加
```

### 1.2 依存関係 (Cargo.toml)

```toml
[dependencies]
tensor4all-treetci = { path = "../tensor4all-treetci" }
```

### 1.3 コールバック型

```rust
/// バッチ評価コールバック
///
/// # Arguments
/// - `batch_data`: column-major (n_sites, n_points) のインデックス配列
///   - batch_data[site + n_sites * point] でアクセス
/// - `n_sites`: サイト数
/// - `n_points`: 評価点数 (n_points=1 のとき point eval 相当)
/// - `results`: 呼び出し側が n_points 個の f64 を書き込む出力バッファ
/// - `user_data`: ユーザーデータポインタ
///
/// # Returns
/// 0 on success, non-zero on error
pub type TreeTciBatchEvalCallback = extern "C" fn(
    batch_data: *const libc::size_t,
    n_sites: libc::size_t,
    n_points: libc::size_t,
    results: *mut libc::c_double,
    user_data: *mut c_void,
) -> i32;
```

Rust 側での closure 変換:

```rust
fn make_batch_eval_closure(
    eval_fn: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
) -> impl Fn(GlobalIndexBatch) -> Result<Vec<f64>> {
    move |batch: GlobalIndexBatch| -> Result<Vec<f64>> {
        let mut results = vec![0.0; batch.n_points()];
        let status = eval_fn(
            batch.data().as_ptr(),
            batch.n_sites(),
            batch.n_points(),
            results.as_mut_ptr(),
            user_data,
        );
        if status != 0 {
            anyhow::bail!("Batch eval callback returned error status {}", status);
        }
        Ok(results)
    }
}

fn make_point_eval_closure(
    eval_fn: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
) -> impl Fn(&[usize]) -> f64 {
    move |indices: &[usize]| -> f64 {
        let mut result: f64 = 0.0;
        let status = eval_fn(
            indices.as_ptr(),
            indices.len(),
            1,
            &mut result,
            user_data,
        );
        if status != 0 { f64::NAN } else { result }
    }
}
```

### 1.4 列挙型

```rust
/// Proposer 選択
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_treetci_proposer_kind {
    /// DefaultProposer: neighbor-product (TreeTCI.jl と同等)
    Default = 0,
    /// SimpleProposer: ランダム (seed ベース)
    Simple = 1,
    /// TruncatedDefaultProposer: Default の truncated ランダムサブセット
    TruncatedDefault = 2,
}
```

`types.rs` に追加し、`From<t4a_treetci_proposer_kind>` で Rust の proposer オブジェクトに変換する。

### 1.5 オペーク型

#### t4a_treetci_graph

```rust
// types.rs に追加
#[repr(C)]
pub struct t4a_treetci_graph {
    pub(crate) _private: *const c_void,
}

impl t4a_treetci_graph {
    pub(crate) fn new(inner: TreeTciGraph) -> Self {
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const c_void,
        }
    }

    pub(crate) fn inner(&self) -> &TreeTciGraph {
        unsafe { &*(self._private as *const TreeTciGraph) }
    }
}

impl Clone for t4a_treetci_graph {
    fn clone(&self) -> Self {
        Self::new(self.inner().clone())
    }
}

impl Drop for t4a_treetci_graph {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe { let _ = Box::from_raw(self._private as *mut TreeTciGraph); }
        }
    }
}

unsafe impl Send for t4a_treetci_graph {}
unsafe impl Sync for t4a_treetci_graph {}
```

#### t4a_treetci_f64

```rust
// types.rs に追加
#[repr(C)]
pub struct t4a_treetci_f64 {
    pub(crate) _private: *const c_void,
}

impl t4a_treetci_f64 {
    pub(crate) fn new(inner: SimpleTreeTci<f64>) -> Self {
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const c_void,
        }
    }

    pub(crate) fn inner(&self) -> &SimpleTreeTci<f64> {
        unsafe { &*(self._private as *const SimpleTreeTci<f64>) }
    }

    pub(crate) fn inner_mut(&mut self) -> &mut SimpleTreeTci<f64> {
        unsafe { &mut *(self._private as *mut SimpleTreeTci<f64>) }
    }
}

impl Drop for t4a_treetci_f64 {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe { let _ = Box::from_raw(self._private as *mut SimpleTreeTci<f64>); }
        }
    }
}

// Clone は実装しない (TCI2 と同様)
unsafe impl Send for t4a_treetci_f64 {}
unsafe impl Sync for t4a_treetci_f64 {}
```

### 1.6 C API 関数一覧

すべて `treetci.rs` に実装。各関数は `catch_unwind(AssertUnwindSafe(...))` でパニック保護。

#### 1.6.1 グラフ ライフサイクル

```rust
impl_opaque_type_common!(treetci_graph);
// 生成: t4a_treetci_graph_release, t4a_treetci_graph_clone, t4a_treetci_graph_is_assigned
```

```rust
/// 木グラフを作成
///
/// # Arguments
/// - `n_sites`: サイト数 (>= 1)
/// - `edges_flat`: エッジ配列 [u0, v0, u1, v1, ...] (length = n_edges * 2)
/// - `n_edges`: エッジ数 (n_sites - 1 であること)
///
/// # Returns
/// 新しい t4a_treetci_graph ポインタ。エラー時は NULL。
/// バリデーション: 連結性、辺数 = n_sites - 1、自己ループなし、重複辺なし
///
/// # Safety
/// edges_flat は n_edges * 2 個の size_t を含む有効なバッファであること
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_graph_new(
    n_sites: libc::size_t,
    edges_flat: *const libc::size_t,
    n_edges: libc::size_t,
) -> *mut t4a_treetci_graph;

/// サイト数を取得
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_graph_n_sites(
    graph: *const t4a_treetci_graph,
    out: *mut libc::size_t,
) -> StatusCode;

/// エッジ数を取得
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_graph_n_edges(
    graph: *const t4a_treetci_graph,
    out: *mut libc::size_t,
) -> StatusCode;
```

#### 1.6.2 ステート ライフサイクル

```rust
/// SimpleTreeTci ステートを作成
///
/// # Arguments
/// - `local_dims`: 各サイトの局所次元 (length = n_sites)
/// - `n_sites`: サイト数 (graph の n_sites と一致すること)
/// - `graph`: 木グラフハンドル (所有権は移転しない、内部で clone)
///
/// # Returns
/// 新しい t4a_treetci_f64 ポインタ。エラー時は NULL。
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_new(
    local_dims: *const libc::size_t,
    n_sites: libc::size_t,
    graph: *const t4a_treetci_graph,
) -> *mut t4a_treetci_f64;

/// ステートを解放
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_release(
    ptr: *mut t4a_treetci_f64,
);
```

#### 1.6.3 ピボット管理

```rust
/// グローバルピボットを追加
///
/// 各ピボットは全サイトのインデックスを持つマルチインデックス。
/// pivots_flat は column-major (n_sites, n_pivots) レイアウト。
///
/// # Arguments
/// - `ptr`: ステートハンドル
/// - `pivots_flat`: column-major (n_sites, n_pivots) のインデックス配列
/// - `n_sites`: サイト数 (ステートの n_sites と一致すること)
/// - `n_pivots`: ピボット数
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_add_global_pivots(
    ptr: *mut t4a_treetci_f64,
    pivots_flat: *const libc::size_t,
    n_sites: libc::size_t,
    n_pivots: libc::size_t,
) -> StatusCode;
```

#### 1.6.4 Sweep 実行

```rust
/// 1イテレーション実行（全辺を1回訪問）
///
/// 内部で AllEdges visitor を使い、指定された proposer で候補を生成し、
/// matrixluci で pivot 選択を行い、ステートを更新する。
///
/// # Arguments
/// - `ptr`: ステートハンドル (mutable)
/// - `eval_cb`: バッチ評価コールバック
/// - `user_data`: コールバックに渡すユーザーデータ
/// - `proposer_kind`: proposer 選択 (Default=0, Simple=1, TruncatedDefault=2)
/// - `tolerance`: このイテレーションの相対 tolerance
/// - `max_bond_dim`: 最大ボンド次元 (0 = 無制限)
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_sweep(
    ptr: *mut t4a_treetci_f64,
    eval_cb: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
    proposer_kind: t4a_treetci_proposer_kind,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
) -> StatusCode;
```

#### 1.6.5 ステート検査

```rust
/// 全辺の最大ボンドエラーを取得
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_max_bond_error(
    ptr: *const t4a_treetci_f64,
    out: *mut libc::c_double,
) -> StatusCode;

/// 最大ボンド次元（現在の最大 rank）を取得
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_max_rank(
    ptr: *const t4a_treetci_f64,
    out: *mut libc::size_t,
) -> StatusCode;

/// 観測された最大サンプル値を取得（正規化用）
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_max_sample_value(
    ptr: *const t4a_treetci_f64,
    out: *mut libc::c_double,
) -> StatusCode;

/// 各辺のボンド次元を取得
///
/// # Arguments
/// - `out_ranks`: 出力バッファ (length >= n_edges)
/// - `buf_len`: バッファサイズ
/// - `out_n_edges`: 実際のエッジ数を出力
///
/// query-then-fill パターン: out_ranks=NULL で out_n_edges のみ取得可能
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_bond_dims(
    ptr: *const t4a_treetci_f64,
    out_ranks: *mut libc::size_t,
    buf_len: libc::size_t,
    out_n_edges: *mut libc::size_t,
) -> StatusCode;
```

#### 1.6.6 Materialization

```rust
/// 収束したステートから TreeTN を構築
///
/// 内部で batch_eval を使ってテンソル値を再評価し、LU 分解で
/// 各辺のテンソルを構築する。
///
/// # Arguments
/// - `ptr`: ステートハンドル (const — ステート自体は変更しない)
/// - `eval_cb`: バッチ評価コールバック (materialization 時にテンソル値を再評価)
/// - `user_data`: コールバックに渡すユーザーデータ
/// - `center_site`: BFS ルートサイト (materialization の中心)
/// - `out_treetn`: 出力 TreeTN ハンドルポインタ
///
/// # Returns
/// 既存の t4a_treetn 型として返す。呼び出し側は t4a_treetn_release で解放。
#[unsafe(no_mangle)]
pub extern "C" fn t4a_treetci_f64_to_treetn(
    ptr: *const t4a_treetci_f64,
    eval_cb: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
    center_site: libc::size_t,
    out_treetn: *mut *mut t4a_treetn,
) -> StatusCode;
```

#### 1.6.7 高レベル便利関数

```rust
/// TreeTCI を一発実行して TreeTN を取得
///
/// 内部で以下を実行:
/// 1. SimpleTreeTci を作成
/// 2. initial_pivots を追加
/// 3. max_iter 回まで sweep を繰り返す (convergence check あり)
/// 4. TreeTN に materialize
///
/// # Arguments
/// - `eval_cb`: バッチ評価コールバック
/// - `user_data`: コールバックに渡すユーザーデータ
/// - `local_dims`: 各サイトの局所次元 (length = n_sites)
/// - `n_sites`: サイト数
/// - `graph`: 木グラフハンドル
/// - `initial_pivots_flat`: column-major (n_sites, n_pivots)、NULL 可 (空ピボット)
/// - `n_pivots`: 初期ピボット数
/// - `proposer_kind`: proposer 選択
/// - `tolerance`: 相対 tolerance
/// - `max_bond_dim`: 最大ボンド次元 (0 = 無制限)
/// - `max_iter`: 最大イテレーション数
/// - `normalize_error`: 正規化フラグ (0=false, 1=true)
/// - `center_site`: materialization の中心サイト
/// - `out_treetn`: 出力 TreeTN ハンドル
/// - `out_ranks`: 各イテレーションの最大 rank (バッファ length >= max_iter, NULL可)
/// - `out_errors`: 各イテレーションの正規化エラー (バッファ length >= max_iter, NULL可)
/// - `out_n_iters`: 実際のイテレーション数
#[unsafe(no_mangle)]
pub extern "C" fn t4a_crossinterpolate_tree_f64(
    eval_cb: TreeTciBatchEvalCallback,
    user_data: *mut c_void,
    local_dims: *const libc::size_t,
    n_sites: libc::size_t,
    graph: *const t4a_treetci_graph,
    initial_pivots_flat: *const libc::size_t,
    n_pivots: libc::size_t,
    proposer_kind: t4a_treetci_proposer_kind,
    tolerance: libc::c_double,
    max_bond_dim: libc::size_t,
    max_iter: libc::size_t,
    normalize_error: libc::c_int,
    center_site: libc::size_t,
    out_treetn: *mut *mut t4a_treetn,
    out_ranks: *mut libc::size_t,
    out_errors: *mut libc::c_double,
    out_n_iters: *mut libc::size_t,
) -> StatusCode;
```

### 1.7 テスト方針

`crates/tensor4all-capi/tests/test_treetci.rs` に統合テスト:

1. **グラフ構築テスト**: 正常ケース + 不正なグラフ (非連結、自己ループ) のバリデーション
2. **ステートフル API テスト**: 7-site tree 上の既知関数で new → add_pivots → sweep → 検査 → to_treetn の全ライフサイクル
3. **高レベル関数テスト**: `crossinterpolate_tree_f64` で同じ既知関数を一発実行し、結果を比較
4. **バッチコールバックテスト**: n_points > 1 のバッチが正しく column-major で渡されることを検証

TreeTCI.jl の既存 parity テストと同じテスト関数 (7-site tree) を使用。

---

## Part 2: Julia ラッパー (Tensor4all.jl 側)

### 2.1 ファイル構成

```
src/
  TreeTCI.jl       # 新規: TreeTCI モジュール
  C_API.jl         # 既存: _sym() に新しい関数名を追加
  Tensor4all.jl    # 既存: include("TreeTCI.jl") を追加
test/
  test_treetci.jl  # 新規: TreeTCI テスト
  runtests.jl      # 既存: test_treetci.jl を include
```

### 2.2 C API バインディング追加 (`src/C_API.jl`)

既存の `_sym(name)` パターンで新しい関数シンボルを解決。追加コードは不要（`_sym` は動的に `dlsym` するため）。

### 2.3 モジュール定義 (`src/TreeTCI.jl`)

```julia
module TreeTCI

using ..Tensor4all: C_API, TreeTN  # 内部依存

export TreeTciGraph, SimpleTreeTci
export crossinterpolate_tree

# ============================================================================
# TreeTciGraph
# ============================================================================

"""
    TreeTciGraph(n_sites, edges)

木グラフ構造を定義する。

# Arguments
- `n_sites::Int`: サイト数
- `edges::Vector{Tuple{Int,Int}}`: エッジのリスト (0-based site indices)

# Example
```julia
# 線形チェーン: 0-1-2-3
graph = TreeTciGraph(4, [(0,1), (1,2), (2,3)])

# スターグラフ: 0 を中心に 1,2,3 が接続
graph = TreeTciGraph(4, [(0,1), (0,2), (0,3)])
```
"""
mutable struct TreeTciGraph
    ptr::Ptr{Cvoid}
    n_sites::Int

    function TreeTciGraph(n_sites::Int, edges::Vector{Tuple{Int,Int}})
        n_edges = length(edges)
        edges_flat = Vector{Csize_t}(undef, n_edges * 2)
        for (i, (u, v)) in enumerate(edges)
            edges_flat[2i - 1] = u
            edges_flat[2i] = v
        end
        ptr = ccall(
            C_API._sym(:t4a_treetci_graph_new),
            Ptr{Cvoid},
            (Csize_t, Ptr{Csize_t}, Csize_t),
            n_sites, edges_flat, n_edges,
        )
        ptr == C_NULL && error("Failed to create TreeTciGraph: $(C_API.last_error())")
        obj = new(ptr, n_sites)
        finalizer(obj) do x
            if x.ptr != C_NULL
                ccall(C_API._sym(:t4a_treetci_graph_release), Cvoid, (Ptr{Cvoid},), x.ptr)
                x.ptr = C_NULL
            end
        end
        obj
    end
end

# ============================================================================
# Batch Eval Trampoline
# ============================================================================

"""
コールバックトランポリン。

C側から呼ばれ、Julia のユーザー関数を呼び出す。
batch_data は column-major (n_sites, n_points)。
ユーザー関数のシグネチャ: f(batch::Matrix{Int}) -> Vector{Float64}
"""
function _treetci_batch_trampoline(
    batch_data::Ptr{Csize_t},
    n_sites::Csize_t,
    n_points::Csize_t,
    results::Ptr{Cdouble},
    user_data::Ptr{Cvoid},
)::Cint
    try
        f_ref = unsafe_pointer_to_objref(user_data)::Ref{Any}
        f = f_ref[]
        # column-major (n_sites, n_points) を Julia Matrix として wrap
        batch = unsafe_wrap(Array, batch_data, (Int(n_sites), Int(n_points)))
        vals = f(batch)
        for i in 1:Int(n_points)
            unsafe_store!(results, vals[i], i)
        end
        return Cint(0)
    catch e
        # エラーを stderr に出力し、非ゼロを返す
        @error "TreeTCI batch eval callback error" exception = (e, catch_backtrace())
        return Cint(-1)
    end
end

# C function pointer (モジュールロード時に一度だけ生成)
const _BATCH_TRAMPOLINE_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _get_batch_trampoline()
    if _BATCH_TRAMPOLINE_PTR[] == C_NULL
        _BATCH_TRAMPOLINE_PTR[] = @cfunction(
            _treetci_batch_trampoline,
            Cint,
            (Ptr{Csize_t}, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cvoid}),
        )
    end
    _BATCH_TRAMPOLINE_PTR[]
end

# ============================================================================
# Proposer 変換
# ============================================================================

const PROPOSER_DEFAULT = Cint(0)
const PROPOSER_SIMPLE = Cint(1)
const PROPOSER_TRUNCATED_DEFAULT = Cint(2)

function _proposer_to_cint(proposer::Symbol)::Cint
    if proposer == :default
        PROPOSER_DEFAULT
    elseif proposer == :simple
        PROPOSER_SIMPLE
    elseif proposer == :truncated_default
        PROPOSER_TRUNCATED_DEFAULT
    else
        error("Unknown proposer: $proposer. Use :default, :simple, or :truncated_default")
    end
end

# ============================================================================
# SimpleTreeTci
# ============================================================================

"""
    SimpleTreeTci(local_dims, graph)

ステートフルな TreeTCI オブジェクト。

# Arguments
- `local_dims::Vector{Int}`: 各サイトの局所次元 (length = graph.n_sites)
- `graph::TreeTciGraph`: 木グラフ構造

# Lifecycle
```julia
tci = SimpleTreeTci(local_dims, graph)
add_global_pivots!(tci, pivots)
for i in 1:max_iter
    sweep!(tci, f; tolerance=1e-8)
    @info "iter \$i" max_bond_error(tci) max_rank(tci)
    max_bond_error(tci) < tolerance && break
end
ttn = to_treetn(tci, f)
```
"""
mutable struct SimpleTreeTci
    ptr::Ptr{Cvoid}
    graph::TreeTciGraph      # GC 保護のため参照保持
    local_dims::Vector{Int}

    function SimpleTreeTci(local_dims::Vector{Int}, graph::TreeTciGraph)
        @assert length(local_dims) == graph.n_sites
        dims_csize = Csize_t.(local_dims)
        ptr = ccall(
            C_API._sym(:t4a_treetci_f64_new),
            Ptr{Cvoid},
            (Ptr{Csize_t}, Csize_t, Ptr{Cvoid}),
            dims_csize, length(dims_csize), graph.ptr,
        )
        ptr == C_NULL && error("Failed to create SimpleTreeTci: $(C_API.last_error())")
        obj = new(ptr, graph, local_dims)
        finalizer(obj) do x
            if x.ptr != C_NULL
                ccall(C_API._sym(:t4a_treetci_f64_release), Cvoid, (Ptr{Cvoid},), x.ptr)
                x.ptr = C_NULL
            end
        end
        obj
    end
end

# ============================================================================
# ピボット管理
# ============================================================================

"""
    add_global_pivots!(tci, pivots)

グローバルピボットを追加する。

# Arguments
- `tci::SimpleTreeTci`: ステート
- `pivots::Vector{Vector{Int}}`: 各ピボットは全サイトのインデックス (0-based)
"""
function add_global_pivots!(tci::SimpleTreeTci, pivots::Vector{Vector{Int}})
    n_sites = length(tci.local_dims)
    n_pivots = length(pivots)
    n_pivots == 0 && return
    # column-major (n_sites, n_pivots) に pack
    pivots_flat = Vector{Csize_t}(undef, n_sites * n_pivots)
    for j in 1:n_pivots
        @assert length(pivots[j]) == n_sites
        for i in 1:n_sites
            pivots_flat[i + n_sites * (j - 1)] = pivots[j][i]
        end
    end
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_add_global_pivots),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Csize_t),
        tci.ptr, pivots_flat, n_sites, n_pivots,
    ))
end

# ============================================================================
# Sweep 実行
# ============================================================================

"""
    sweep!(tci, f; proposer=:default, tolerance=1e-8, max_bond_dim=0)

1イテレーション（全辺を1回訪問）を実行する。

# Arguments
- `tci::SimpleTreeTci`: ステート
- `f`: 評価関数 `f(batch::Matrix{Int}) -> Vector{Float64}`
  - `batch` は column-major (n_sites, n_points), 0-based indices
  - 戻り値は n_points 個の Float64
- `proposer`: `:default`, `:simple`, `:truncated_default`
- `tolerance`: 相対 tolerance
- `max_bond_dim`: 最大ボンド次元 (0 = 無制限)
"""
function sweep!(tci::SimpleTreeTci, f;
    proposer::Symbol = :default,
    tolerance::Float64 = 1e-8,
    max_bond_dim::Int = 0,
)
    f_ref = Ref{Any}(f)
    GC.@preserve f_ref begin
        C_API.check_status(ccall(
            C_API._sym(:t4a_treetci_f64_sweep),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cdouble, Csize_t),
            tci.ptr,
            _get_batch_trampoline(),
            pointer_from_objref(f_ref),
            _proposer_to_cint(proposer),
            tolerance,
            max_bond_dim,
        ))
    end
end

# ============================================================================
# ステート検査
# ============================================================================

"""全辺の最大ボンドエラーを取得"""
function max_bond_error(tci::SimpleTreeTci)::Float64
    out = Ref{Cdouble}(0.0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_max_bond_error),
        Cint, (Ptr{Cvoid}, Ptr{Cdouble}),
        tci.ptr, out,
    ))
    out[]
end

"""最大ボンド次元を取得"""
function max_rank(tci::SimpleTreeTci)::Int
    out = Ref{Csize_t}(0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_max_rank),
        Cint, (Ptr{Cvoid}, Ptr{Csize_t}),
        tci.ptr, out,
    ))
    Int(out[])
end

"""観測された最大サンプル値を取得"""
function max_sample_value(tci::SimpleTreeTci)::Float64
    out = Ref{Cdouble}(0.0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_max_sample_value),
        Cint, (Ptr{Cvoid}, Ptr{Cdouble}),
        tci.ptr, out,
    ))
    out[]
end

"""各辺のボンド次元を取得"""
function bond_dims(tci::SimpleTreeTci)::Vector{Int}
    # query: エッジ数を取得
    n_edges_ref = Ref{Csize_t}(0)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_bond_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
        tci.ptr, C_NULL, 0, n_edges_ref,
    ))
    n_edges = Int(n_edges_ref[])
    # fill: バッファに書き込み
    buf = Vector{Csize_t}(undef, n_edges)
    C_API.check_status(ccall(
        C_API._sym(:t4a_treetci_f64_bond_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ptr{Csize_t}),
        tci.ptr, buf, n_edges, n_edges_ref,
    ))
    Int.(buf)
end

# ============================================================================
# Materialization
# ============================================================================

"""
    to_treetn(tci, f; center_site=0)

収束した TreeTCI ステートから TreeTensorNetwork を構築する。

# Arguments
- `tci::SimpleTreeTci`: 収束したステート
- `f`: 評価関数 (sweep! と同じシグネチャ)
- `center_site`: materialization の中心サイト (0-based)

# Returns
- `TreeTensorNetwork`: 既存の TreeTN ラッパー型
"""
function to_treetn(tci::SimpleTreeTci, f; center_site::Int = 0)
    f_ref = Ref{Any}(f)
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve f_ref begin
        C_API.check_status(ccall(
            C_API._sym(:t4a_treetci_f64_to_treetn),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{Ptr{Cvoid}}),
            tci.ptr,
            _get_batch_trampoline(),
            pointer_from_objref(f_ref),
            center_site,
            out_ptr,
        ))
    end
    TreeTN.TreeTensorNetwork(out_ptr[])
end

# ============================================================================
# 高レベル便利関数
# ============================================================================

"""
    crossinterpolate_tree(f, local_dims, graph; kwargs...) -> (ttn, ranks, errors)

TreeTCI を一発実行して TreeTensorNetwork を取得する。

# Arguments
- `f`: 評価関数 `f(batch::Matrix{Int}) -> Vector{Float64}`
- `local_dims::Vector{Int}`: 各サイトの局所次元
- `graph::TreeTciGraph`: 木グラフ構造

# Keyword Arguments
- `initial_pivots::Vector{Vector{Int}} = []`: 初期ピボット (0-based)
- `proposer::Symbol = :default`: proposer 選択
- `tolerance::Float64 = 1e-8`: 相対 tolerance
- `max_bond_dim::Int = 0`: 最大ボンド次元 (0=無制限)
- `max_iter::Int = 20`: 最大イテレーション数
- `normalize_error::Bool = true`: エラー正規化フラグ
- `center_site::Int = 0`: materialization の中心サイト

# Returns
- `ttn::TreeTensorNetwork`: 結果のテンソルネットワーク
- `ranks::Vector{Int}`: 各イテレーションの最大 rank
- `errors::Vector{Float64}`: 各イテレーションの正規化エラー
"""
function crossinterpolate_tree(
    f, local_dims::Vector{Int}, graph::TreeTciGraph;
    initial_pivots::Vector{Vector{Int}} = Vector{Int}[],
    proposer::Symbol = :default,
    tolerance::Float64 = 1e-8,
    max_bond_dim::Int = 0,
    max_iter::Int = 20,
    normalize_error::Bool = true,
    center_site::Int = 0,
)
    n_sites = length(local_dims)
    n_pivots = length(initial_pivots)

    # initial_pivots を column-major (n_sites, n_pivots) に pack
    pivots_flat = if n_pivots > 0
        buf = Vector{Csize_t}(undef, n_sites * n_pivots)
        for j in 1:n_pivots
            for i in 1:n_sites
                buf[i + n_sites * (j - 1)] = initial_pivots[j][i]
            end
        end
        buf
    else
        Csize_t[]
    end

    # 出力バッファ (max_iter 分を事前確保)
    out_ranks = Vector{Csize_t}(undef, max_iter)
    out_errors = Vector{Cdouble}(undef, max_iter)
    out_n_iters = Ref{Csize_t}(0)
    out_treetn = Ref{Ptr{Cvoid}}(C_NULL)

    f_ref = Ref{Any}(f)
    GC.@preserve f_ref pivots_flat out_ranks out_errors begin
        C_API.check_status(ccall(
            C_API._sym(:t4a_crossinterpolate_tree_f64),
            Cint,
            (
                Ptr{Cvoid}, Ptr{Cvoid},                          # eval_cb, user_data
                Ptr{Csize_t}, Csize_t,                           # local_dims, n_sites
                Ptr{Cvoid},                                       # graph
                Ptr{Csize_t}, Csize_t,                           # initial_pivots, n_pivots
                Cint,                                             # proposer_kind
                Cdouble, Csize_t, Csize_t,                       # tol, max_bond_dim, max_iter
                Cint,                                             # normalize_error
                Csize_t,                                          # center_site
                Ptr{Ptr{Cvoid}},                                  # out_treetn
                Ptr{Csize_t}, Ptr{Cdouble}, Ptr{Csize_t},       # out_ranks, errors, n_iters
            ),
            _get_batch_trampoline(),
            pointer_from_objref(f_ref),
            Csize_t.(local_dims), n_sites,
            graph.ptr,
            n_pivots > 0 ? pivots_flat : C_NULL, n_pivots,
            _proposer_to_cint(proposer),
            tolerance, max_bond_dim, max_iter,
            normalize_error ? Cint(1) : Cint(0),
            center_site,
            out_treetn,
            out_ranks, out_errors, out_n_iters,
        ))
    end

    n_iters = Int(out_n_iters[])
    ttn = TreeTN.TreeTensorNetwork(out_treetn[])
    ranks = Int.(out_ranks[1:n_iters])
    errors = Float64.(out_errors[1:n_iters])
    return (ttn, ranks, errors)
end

end # module TreeTCI
```

### 2.4 メインモジュールへの統合 (`src/Tensor4all.jl`)

```julia
# 既存の include の後に追加
include("TreeTCI.jl")
```

### 2.5 テスト (`test/test_treetci.jl`)

```julia
using Tensor4all.TreeTCI
using Test

@testset "TreeTCI" begin
    @testset "TreeTciGraph" begin
        # 線形チェーン
        graph = TreeTciGraph(4, [(0,1), (1,2), (2,3)])
        # スターグラフ
        graph_star = TreeTciGraph(4, [(0,1), (0,2), (0,3)])
        # 不正なグラフはエラー
        @test_throws ErrorException TreeTciGraph(4, [(0,1), (2,3)])  # 非連結
    end

    @testset "Stateful API" begin
        # 7-site star tree (TreeTCI.jl parity test に相当)
        n_sites = 7
        local_dims = fill(2, n_sites)
        edges = [(0,i) for i in 1:6]  # site 0 を中心としたスター
        graph = TreeTciGraph(n_sites, edges)

        # テスト関数: 全サイトの積
        function f_batch(batch::Matrix{<:Integer})
            n_pts = size(batch, 2)
            results = Vector{Float64}(undef, n_pts)
            for j in 1:n_pts
                val = 1.0
                for i in 1:size(batch, 1)
                    val *= (batch[i, j] + 1.0)
                end
                results[j] = val
            end
            results
        end

        tci = SimpleTreeTci(local_dims, graph)
        add_global_pivots!(tci, [zeros(Int, n_sites)])  # all-zero pivot

        for iter in 1:10
            sweep!(tci, f_batch; tolerance=1e-12)
            max_bond_error(tci) < 1e-12 && break
        end

        @test max_bond_error(tci) < 1e-10
        @test max_rank(tci) >= 1

        ttn = to_treetn(tci, f_batch)
        # ttn を使って既存の TreeTN API で検証可能
    end

    @testset "High-level API" begin
        n_sites = 4
        local_dims = fill(3, n_sites)
        graph = TreeTciGraph(n_sites, [(0,1), (1,2), (2,3)])

        function f_batch(batch)
            [sum(Float64, batch[:, j]) for j in 1:size(batch, 2)]
        end

        ttn, ranks, errors = crossinterpolate_tree(
            f_batch, local_dims, graph;
            initial_pivots = [zeros(Int, n_sites)],
            tolerance = 1e-10,
            max_iter = 20,
        )

        @test length(ranks) > 0
        @test last(errors) < 1e-8
    end
end
```

---

## まとめ: 実装順序

| Step | リポジトリ | 内容 |
|------|-----------|------|
| 1 | tensor4all-rs | `Cargo.toml` に `tensor4all-treetci` 依存追加 |
| 2 | tensor4all-rs | `types.rs` に `t4a_treetci_graph`, `t4a_treetci_f64`, `t4a_treetci_proposer_kind` 追加 |
| 3 | tensor4all-rs | `treetci.rs` 新規作成: 全 C API 関数実装 |
| 4 | tensor4all-rs | `lib.rs` に `mod treetci; pub use treetci::*;` 追加 |
| 5 | tensor4all-rs | `tests/test_treetci.rs` で C API テスト |
| 6 | Tensor4all.jl | `src/TreeTCI.jl` 新規作成 |
| 7 | Tensor4all.jl | `src/Tensor4all.jl` に include 追加 |
| 8 | Tensor4all.jl | `test/test_treetci.jl` で統合テスト |
