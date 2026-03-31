# Complex64 対応 & Interleaved 統一 設計

## Overview

tensor4all-rs C API と Tensor4all.jl において、Complex64 (c64) サポートを追加し、
complex データ表現を **interleaved** (`[re0, im0, re1, im1, ...]`) に統一する。

## 背景

- Rust 側は `TensorTrain<Complex64>`, `TensorCI2<Complex64>`, `SimpleTreeTci<Complex64>` を
  ジェネリクスで完全サポート済み
- C API は現在 f64 のみ。唯一の complex 対応 (`t4a_tensor_new_dense_c64`) は
  **separated buffers** (re配列 + im配列) だが、これは不自然
- Julia `ComplexF64`, C `double _Complex`, Rust `Complex64` は全てメモリ上 `[re, im]` の
  interleaved なので、separated は変換コストが無駄

## 設計方針

### Interleaved 表現

Complex データは全て `*const f64` / `*mut f64` バッファで、長さ `2 * n_elements`。
`data[2*i]` = real part, `data[2*i+1]` = imaginary part。

Julia 側は `reinterpret(Float64, ::Vector{ComplexF64})` で zero-copy 変換可能。

### C API の命名規則

既存パターンに従い `_f64` / `_c64` サフィックスで区別:

```
t4a_simplett_f64_evaluate  →  t4a_simplett_c64_evaluate
t4a_tci2_f64_sweep2site   →  t4a_tci2_c64_sweep2site
t4a_treetci_f64_sweep      →  t4a_treetci_c64_sweep
```

---

## Part 1: C API 変更 (tensor4all-rs)

### 1.1 既存 tensor API: separated → interleaved 移行

**変更対象:**
- `t4a_tensor_new_dense_c64`: `(data_re, data_im, data_len)` → `(data_interleaved, data_len)`
  - `data_interleaved`: `*const f64`, length = `2 * n_elements`
  - `data_len`: 要素数（complex 要素数, interleaved 配列長の半分）
- `t4a_tensor_get_data_c64`: `(buf_re, buf_im, buf_len, out_len)` → `(buf, buf_len, out_len)`
  - `buf`: `*mut f64`, length = `2 * n_elements`
  - `buf_len`, `out_len`: complex 要素数

**破壊的変更**: はい。Python ラッパーは disable のため問題なし。

### 1.2 新規: SimpleTT c64

**新規オペーク型** (`types.rs`):
```rust
pub struct t4a_simplett_c64 {
    pub(crate) _private: *const c_void,
}
// inner: TensorTrain<Complex64>
// Clone, Drop, Send+Sync 実装
```

**新規関数** (`simplett.rs` に追加):

| 関数 | 説明 |
|------|------|
| `t4a_simplett_c64_release` | 解放 |
| `t4a_simplett_c64_clone` | 複製 |
| `t4a_simplett_c64_constant(site_dims, n_sites, value_re, value_im)` | 定数 TT 作成 |
| `t4a_simplett_c64_zeros(site_dims, n_sites)` | ゼロ TT 作成 |
| `t4a_simplett_c64_len(ptr, out)` | サイト数 |
| `t4a_simplett_c64_site_dims(ptr, buf, ...)` | サイト次元 |
| `t4a_simplett_c64_link_dims(ptr, buf, ...)` | ボンド次元 |
| `t4a_simplett_c64_rank(ptr, out)` | 最大ランク |
| `t4a_simplett_c64_evaluate(ptr, indices, n, out_re, out_im)` | 評価 → complex 値 |
| `t4a_simplett_c64_sum(ptr, out_re, out_im)` | 総和 → complex 値 |
| `t4a_simplett_c64_norm(ptr, out)` | ノルム → f64 |
| `t4a_simplett_c64_site_tensor(ptr, site, buf, buf_len, out_len, out_dims, ...)` | サイトテンソル → interleaved |
| `t4a_simplett_c64_compress(ptr, method, tol, max_bonddim)` | 圧縮 |
| `t4a_simplett_c64_partial_sum(ptr, dims, n_dims, out)` | 部分和 |

**スカラー戻り値**: `evaluate`, `sum` は `out_re: *mut f64, out_im: *mut f64` の2引数で返す。
**テンソルデータ**: `site_tensor` は interleaved buffer。

### 1.3 新規: TensorCI2 c64

**新規コールバック型:**
```rust
/// Complex evaluation callback.
/// result: interleaved [re, im] (2 doubles per point)
pub type EvalCallbackC64 = extern "C" fn(
    indices: *const i64,
    n_indices: libc::size_t,
    result_re: *mut f64,
    result_im: *mut f64,
    user_data: *mut c_void,
) -> i32;
```

注: 単一値の戻りなので `result_re, result_im` の2ポインタ。
バッチではなく1点評価のため interleaved バッファではなく分離で OK。

**新規オペーク型:** `t4a_tci2_c64` (wraps `TensorCI2<Complex64>`)

**新規関数** (`tensorci.rs` に追加):

f64 版と同じ 18 関数 + 高レベル関数:

| 関数 | 差異 |
|------|------|
| `t4a_tci2_c64_new` | 同じ |
| `t4a_tci2_c64_release` | 同じ |
| `t4a_tci2_c64_sweep2site(ptr, eval_cb, user_data, ...)` | EvalCallbackC64 使用 |
| `t4a_tci2_c64_sweep1site(ptr, eval_cb, user_data, ...)` | EvalCallbackC64 使用 |
| `t4a_tci2_c64_fill_site_tensors(ptr, eval_cb, user_data)` | EvalCallbackC64 使用 |
| `t4a_tci2_c64_to_tensor_train(ptr, out)` | `*mut *mut t4a_simplett_c64` |
| `t4a_crossinterpolate2_c64(...)` | EvalCallbackC64 + `*mut *mut t4a_tci2_c64` |
| その他アクセサ | f64 版と同じシグネチャ（rank, link_dims, errors は f64 を返す） |

### 1.4 新規: TreeTCI c64

**新規コールバック型:**
```rust
/// Complex batch evaluation callback.
/// results: interleaved [re0, im0, re1, im1, ...] (2 * n_points doubles)
pub type TreeTciBatchEvalCallbackC64 = extern "C" fn(
    batch_data: *const libc::size_t,
    n_sites: libc::size_t,
    n_points: libc::size_t,
    results: *mut libc::c_double,  // interleaved, length = 2 * n_points
    user_data: *mut c_void,
) -> i32;
```

バッチなので interleaved buffer を使う。

**新規オペーク型:** `t4a_treetci_c64` (wraps `SimpleTreeTci<Complex64>`)

**新規関数** (`treetci.rs` に追加):

| 関数 | 差異 |
|------|------|
| `t4a_treetci_c64_new` | 同じ |
| `t4a_treetci_c64_release` | 同じ |
| `t4a_treetci_c64_add_global_pivots` | 同じ |
| `t4a_treetci_c64_sweep(ptr, eval_cb, ...)` | TreeTciBatchEvalCallbackC64 |
| `t4a_treetci_c64_max_bond_error` | 同じ (f64 戻り) |
| `t4a_treetci_c64_max_rank` | 同じ |
| `t4a_treetci_c64_max_sample_value` | 同じ (f64 戻り) |
| `t4a_treetci_c64_bond_dims` | 同じ |
| `t4a_treetci_c64_to_treetn(ptr, eval_cb, ..., out_treetn)` | TreeTciBatchEvalCallbackC64、出力は `t4a_treetn` |
| `t4a_crossinterpolate_tree_c64(...)` | TreeTciBatchEvalCallbackC64 |

**Graph / Options / Proposer は f64/c64 共通** — 型に依存しないので既存のものを再利用。

### 1.5 テスト方針 (C API)

各モジュールに c64 テストを追加:

- **tensor**: interleaved 版 `new_dense_c64` / `get_data_c64` のラウンドトリップ
- **simplett_c64**: complex constant TT の作成・評価・sum・norm・圧縮
- **tci2_c64**: complex product function `f(idx) = prod((idx[s]+1) + i*(2*idx[s]+1))` でTCI実行
- **treetci_c64**: 同じ complex product function で7-site branching tree、Rust parity テストと同じ tolerance (1e-12)

---

## Part 2: Julia ラッパー変更 (Tensor4all.jl)

### 2.1 Tensor: interleaved 対応

**変更:** `src/Tensor4all.jl` の `Tensor(inds, data::ComplexF64)` コンストラクタと `data()` アクセサ。

```julia
# Before (separated)
data_re = Cdouble[real(z) for z in flat_data]
data_im = Cdouble[imag(z) for z in flat_data]
ptr = C_API.t4a_tensor_new_dense_c64(r, index_ptrs, dims_vec, data_re, data_im)

# After (interleaved)
data_interleaved = reinterpret(Cdouble, ComplexF64.(vec(data)))
ptr = C_API.t4a_tensor_new_dense_c64(r, index_ptrs, dims_vec, data_interleaved)
```

取得側:
```julia
# Before
buf_re = Vector{Cdouble}(undef, n)
buf_im = Vector{Cdouble}(undef, n)
C_API.t4a_tensor_get_data_c64(ptr, buf_re, buf_im, n, out_len)
buf = [ComplexF64(r, i) for (r, i) in zip(buf_re, buf_im)]

# After
buf = Vector{Cdouble}(undef, 2 * n)
C_API.t4a_tensor_get_data_c64(ptr, buf, n, out_len)
result = reinterpret(ComplexF64, buf)
```

### 2.2 SimpleTT: c64 対応

`src/SimpleTT.jl` の `SimpleTensorTrain` を `SimpleTensorTrain{T}` にジェネリック化:

```julia
mutable struct SimpleTensorTrain{T<:Union{Float64, ComplexF64}}
    ptr::Ptr{Cvoid}
end
```

各メソッドで `T` に応じて `_f64` / `_c64` の C API 関数を dispatch:

```julia
function _sym_for(::Type{Float64}, name::Symbol) = C_API._sym(Symbol("t4a_simplett_f64_", name))
function _sym_for(::Type{ComplexF64}, name::Symbol) = C_API._sym(Symbol("t4a_simplett_c64_", name))
```

`evaluate` の戻り値が `T` になる:
- `Float64`: `Ref{Cdouble}` → `Float64`
- `ComplexF64`: `Ref{Cdouble}` × 2 (re, im) → `ComplexF64`

### 2.3 TensorCI: c64 対応

`src/TensorCI.jl` の `TensorCI2{T}` を拡張:

```julia
mutable struct TensorCI2{T<:Union{Float64, ComplexF64}}
    ptr::Ptr{Cvoid}
    local_dims::Vector{Int}
end
```

コールバック trampoline を追加:
- f64: 既存 `_trampoline` (変更なし)
- c64: 新規 `_trampoline_c64` — Julia 関数が `ComplexF64` を返し、re/im に分離

```julia
function _trampoline_c64(indices_ptr, n_indices, result_re, result_im, user_data)::Cint
    f = unsafe_pointer_to_objref(user_data)::Ref{Any} |> x -> x[]
    indices = unsafe_wrap(Array, indices_ptr, Int(n_indices))
    val = ComplexF64(f(indices...))
    unsafe_store!(result_re, real(val))
    unsafe_store!(result_im, imag(val))
    Cint(0)
end
```

### 2.4 TreeTCI: c64 対応

`src/TreeTCI.jl` に c64 対応を追加:

- `SimpleTreeTci{T}` にジェネリック化
- バッチ trampoline:
  - f64: 既存 (変更なし)
  - c64: Julia 関数が `Vector{ComplexF64}` を返し、`reinterpret(Float64, vals)` で interleaved に変換

```julia
function _treetci_batch_trampoline_c64(batch_data, n_sites, n_points, results, user_data)::Cint
    f = unsafe_pointer_to_objref(user_data)::Ref{Any} |> x -> x[]
    batch = unsafe_wrap(Array, batch_data, (Int(n_sites), Int(n_points)))
    vals = ComplexF64.(f(batch))
    interleaved = reinterpret(Float64, vals)
    for i in eachindex(interleaved)
        unsafe_store!(results, interleaved[i], i)
    end
    Cint(0)
end
```

`crossinterpolate_tree` も `T` パラメータで dispatch。

### 2.5 テスト方針 (Julia)

既存の f64 テストに加え、各モジュールに c64 テストを追加:

- **tensor**: ComplexF64 ラウンドトリップ（interleaved 版）
- **simplett_c64**: complex constant TT 作成・評価
- **tci2_c64**: complex product function でTCI
- **treetci_c64**: complex product function で7-site branching tree

Rust parity テストと **同一の関数・パラメータ・tolerance** を使用。

---

## 実装順序

| Step | リポジトリ | 内容 |
|------|-----------|------|
| 1 | tensor4all-rs | tensor API: separated → interleaved 移行 |
| 2 | tensor4all-rs | simplett_c64 追加 |
| 3 | tensor4all-rs | tci2_c64 追加 |
| 4 | tensor4all-rs | treetci_c64 追加 |
| 5 | Tensor4all.jl | tensor interleaved 対応 |
| 6 | Tensor4all.jl | SimpleTT{T} ジェネリック化 |
| 7 | Tensor4all.jl | TensorCI{T} ジェネリック化 |
| 8 | Tensor4all.jl | TreeTCI{T} ジェネリック化 |

各ステップは独立した PR にできる。Step 1-4 は tensor4all-rs 内で1つの PR にまとめても可。

---

## 設計上の注意

1. **Graph / Options / Proposer は共通** — complex で変わらない
2. **norm, max_bond_error, max_sample_value は常に f64** — complex でも実数値
3. **evaluate, sum の戻り値は T** — complex の場合は `(re, im)` ペアで返す
4. **site_tensor データは interleaved** — `length = 2 * n_elements` の f64 バッファ
5. **Python ラッパーは無視** — disable 方針
