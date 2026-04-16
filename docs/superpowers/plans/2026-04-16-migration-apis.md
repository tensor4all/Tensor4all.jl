# Phase 1: Core Migration APIs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement tensor factorization (SVD, QR), TensorTrain orthogonalize/truncate/queries, dag, and QuanticsTransform operator materialization for BubbleTeaCI migration.

**Architecture:** Two-repo. tensor4all-rs gets `t4a_tensor_svd` and `t4a_tensor_qr`. Tensor4all.jl gets Julia wrappers for all new operations plus QuanticsTransform materialization via existing `t4a_qtransform_*` C API.

**Tech Stack:** Rust (tensor4all-rs), Julia (Tensor4all.jl), C FFI

**Spec:** `docs/superpowers/specs/2026-04-16-bubbleteaci-migration-apis-design.md`

---

## File Map

### Rust files (tensor4all-rs, separate PR)
- Modify: `crates/tensor4all-capi/src/tensor.rs` — add svd/qr C API functions
- Modify: `crates/tensor4all-capi/include/tensor4all_capi.h` — add declarations
- Create: `crates/tensor4all-capi/src/tensor/tests/svd_qr.rs` — C API tests

### Julia files to create
- `test/core/tensor_factorize.jl` — SVD, QR, dag, Array tests
- `test/tensornetworks/orthogonalize_truncate.jl` — orthogonalize, truncate tests
- `test/tensornetworks/queries.jl` — linkinds, linkdims, siteinds tests
- `test/quanticstransform/materialize.jl` — operator materialization tests

### Julia files to modify
- `src/Core/Tensor.jl` — add dag, svd, qr, Array
- `src/TensorNetworks/backend/treetn.jl` — add orthogonalize, truncate wrappers
- `src/TensorNetworks.jl` — add exports (dag, orthogonalize, truncate, linkinds, etc.)
- `src/TensorNetworks/types.jl` or new file — add linkinds, linkdims, siteinds
- `src/QuanticsTransform.jl` — replace placeholders with real materialization
- `src/TensorNetworks/backend/capi.jl` — add QTT layout constants if needed
- `src/Tensor4all.jl` — add exports (dag, svd, qr, orthogonalize, truncate, etc.)
- `test/runtests.jl` — include new test files

---

## Task 1: Rust C API — t4a_tensor_svd and t4a_tensor_qr

**Repo:** tensor4all-rs (separate PR, must merge first)

**Files:**
- Modify: `crates/tensor4all-capi/src/tensor.rs`
- Modify: `crates/tensor4all-capi/include/tensor4all_capi.h`
- Modify: `crates/tensor4all-capi/src/tensor/tests/mod.rs`

- [ ] **Step 1: Add t4a_tensor_svd to C API**

In `crates/tensor4all-capi/src/tensor.rs`, add:

```rust
#[no_mangle]
pub extern "C" fn t4a_tensor_svd(
    tensor: *const t4a_tensor,
    left_inds: *const *const t4a_index,
    n_left: usize,
    rtol: libc::c_double,
    cutoff: libc::c_double,
    maxdim: usize,
    out_u: *mut *mut t4a_tensor,
    out_s: *mut *mut t4a_tensor,
    out_v: *mut *mut t4a_tensor,
) -> StatusCode {
    run_status(|| {
        // Extract tensor, collect left indices, call tensor4all_core::svd::svd_with
        // Build SvdOptions from rtol/cutoff/maxdim
        // Return U, S, V as new tensor handles
        // (Implementation delegates to existing tensor4all_core::svd)
    })
}
```

- [ ] **Step 2: Add t4a_tensor_qr to C API**

```rust
#[no_mangle]
pub extern "C" fn t4a_tensor_qr(
    tensor: *const t4a_tensor,
    left_inds: *const *const t4a_index,
    n_left: usize,
    out_q: *mut *mut t4a_tensor,
    out_r: *mut *mut t4a_tensor,
) -> StatusCode {
    run_status(|| {
        // Extract tensor, collect left indices, call tensor4all_core::qr
        // Return Q, R as new tensor handles
    })
}
```

- [ ] **Step 3: Add header declarations**

In `crates/tensor4all-capi/include/tensor4all_capi.h`:

```c
StatusCode t4a_tensor_svd(const struct t4a_tensor *tensor,
                          const struct t4a_index *const *left_inds,
                          size_t n_left,
                          double rtol,
                          double cutoff,
                          size_t maxdim,
                          struct t4a_tensor **out_u,
                          struct t4a_tensor **out_s,
                          struct t4a_tensor **out_v);

StatusCode t4a_tensor_qr(const struct t4a_tensor *tensor,
                         const struct t4a_index *const *left_inds,
                         size_t n_left,
                         struct t4a_tensor **out_q,
                         struct t4a_tensor **out_r);
```

- [ ] **Step 4: Add tests**

In `crates/tensor4all-capi/src/tensor/tests/mod.rs`:

```rust
#[test]
fn test_tensor_svd_rank2() {
    // Create 3x4 matrix tensor, SVD, verify U*S*V† reconstructs
}

#[test]
fn test_tensor_svd_truncation() {
    // Create rank-deficient matrix, SVD with maxdim=1, verify rank reduction
}

#[test]
fn test_tensor_qr_rank2() {
    // Create 3x4 matrix tensor, QR, verify Q*R reconstructs
}
```

- [ ] **Step 5: Verify**

```bash
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release --workspace
```

- [ ] **Step 6: Commit**

```bash
git add crates/tensor4all-capi/
git commit -m "feat(capi): add t4a_tensor_svd and t4a_tensor_qr"
```

---

## Task 2: Tensor dag and Array

**Files:**
- Modify: `src/Core/Tensor.jl`
- Create: `test/core/tensor_factorize.jl`
- Modify: `test/runtests.jl`
- Modify: `src/Tensor4all.jl`

- [ ] **Step 1: Write failing tests**

Create `test/core/tensor_factorize.jl`:

```julia
using Test
using Tensor4all
using LinearAlgebra: norm

@testset "Tensor dag" begin
    i = Index(2; tags=["i"])
    j = Index(3; tags=["j"])
    data = reshape(Complex{Float64}[1+2im, 3+4im, 5+6im, 7+8im, 9+10im, 11+12im], 2, 3)
    t = Tensor(data, [i, j])

    d = dag(t)
    @test inds(d) == [i, j]
    @test d.data ≈ conj(data)

    # dag of real tensor is identity
    t_real = Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j])
    @test dag(t_real).data ≈ t_real.data
end

@testset "Tensor Array with index reordering" begin
    i = Index(2; tags=["i"])
    j = Index(3; tags=["j"])
    data = reshape(collect(1.0:6.0), 2, 3)
    t = Tensor(data, [i, j])

    # Same order
    arr = Array(t, i, j)
    @test arr ≈ data

    # Permuted order
    arr_perm = Array(t, j, i)
    @test arr_perm ≈ permutedims(data, (2, 1))
    @test size(arr_perm) == (3, 2)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `T4A_SKIP_HDF5_TESTS=1 julia --startup-file=no --project=. test/runtests.jl`
Expected: FAIL — `dag` not defined

- [ ] **Step 3: Implement dag and Array**

Add to `src/Core/Tensor.jl` after the `isapprox` definition:

```julia
"""
    dag(t)

Return the conjugate of tensor `t`. Indices are unchanged (no arrow system).
"""
dag(t::Tensor) = Tensor(conj(t.data), inds(t); backend_handle=nothing)

"""
    Array(t, inds...)

Return the dense data of `t` permuted to the requested index order.
"""
function Base.Array(t::Tensor, requested_inds::Index...)
    perm = _match_index_permutation(inds(t), collect(requested_inds))
    return perm == Tuple(1:rank(t)) ? copy(t.data) : permutedims(t.data, perm)
end
```

- [ ] **Step 4: Export dag**

Add to `src/Tensor4all.jl` exports:

```julia
export dag
```

- [ ] **Step 5: Add test include to runtests.jl**

Add to `test/runtests.jl`:

```julia
include("core/tensor_factorize.jl")
```

- [ ] **Step 6: Run tests**

Run: `T4A_SKIP_HDF5_TESTS=1 julia --startup-file=no --project=. test/runtests.jl`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/Core/Tensor.jl src/Tensor4all.jl test/core/tensor_factorize.jl test/runtests.jl
git commit -m "feat: add Tensor dag and Array with index reordering"
```

---

## Task 3: Tensor SVD via C API

**Prerequisite:** tensor4all-rs PR with `t4a_tensor_svd` merged, pin updated.

**Files:**
- Modify: `src/Core/Tensor.jl`
- Modify: `test/core/tensor_factorize.jl`

- [ ] **Step 1: Write failing tests**

Add to `test/core/tensor_factorize.jl`:

```julia
@testset "Tensor SVD" begin
    i = Index(3; tags=["i"])
    j = Index(4; tags=["j"])
    data = reshape(collect(1.0:12.0), 3, 4)
    t = Tensor(data, [i, j])

    @testset "basic SVD" begin
        U, S, V = svd(t, [i])
        # Reconstruct: contract U*S then result*dag(V)
        US = contract(U, S)
        reconstructed = contract(US, dag(V))
        @test isapprox(t, reconstructed; atol=1e-12)
    end

    @testset "vararg form" begin
        U, S, V = svd(t, i)
        US = contract(U, S)
        reconstructed = contract(US, dag(V))
        @test isapprox(t, reconstructed; atol=1e-12)
    end

    @testset "SVD with truncation" begin
        # Rank-1 matrix + noise
        v1 = collect(1.0:3.0)
        v2 = collect(1.0:4.0)
        data_r1 = v1 * v2' + 1e-10 * randn(3, 4)
        t_r1 = Tensor(data_r1, [i, j])
        U, S, V = svd(t_r1, [i]; maxdim=1)
        @test dims(S) == (1, 1)
    end

    @testset "rank-3 tensor SVD" begin
        k = Index(2; tags=["k"])
        data3 = reshape(collect(1.0:24.0), 3, 4, 2)
        t3 = Tensor(data3, [i, j, k])
        U, S, V = svd(t3, [i, j])
        US = contract(U, S)
        reconstructed = contract(US, dag(V))
        @test isapprox(t3, reconstructed; atol=1e-10)
    end

    @testset "error: empty left_inds" begin
        @test_throws ArgumentError svd(t, Index[])
    end

    @testset "error: non-member left_inds" begin
        k = Index(5; tags=["k"])
        @test_throws ArgumentError svd(t, [k])
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `svd` not defined for Tensor

- [ ] **Step 3: Implement SVD**

Add to `src/Core/Tensor.jl`:

```julia
import LinearAlgebra: svd

"""
    svd(t, left_inds::Vector{Index}; rtol=0.0, cutoff=0.0, maxdim=0)

Compute truncated SVD of tensor `t`, grouping `left_inds` on the left.
Returns `(U, S, V)` as Tensors. Reconstruct via `contract(contract(U, S), dag(V))`.
"""
function LinearAlgebra.svd(
    t::Tensor, left_inds::Vector{Index};
    rtol::Real=0.0, cutoff::Real=0.0, maxdim::Integer=0,
)
    isempty(left_inds) && throw(ArgumentError("left_inds must not be empty"))
    for idx in left_inds
        idx in inds(t) || throw(ArgumentError("Index $idx not found in tensor indices $(inds(t))"))
    end
    length(left_inds) == rank(t) && throw(ArgumentError("left_inds must not contain all indices"))

    scalar_kind = _tensor_scalar_kind(t)
    tn = _tensor_networks_module()
    t_handle = C_NULL
    left_handles = Ptr{Cvoid}[]
    u_handle = C_NULL
    s_handle = C_NULL
    v_handle = C_NULL

    try
        t_handle = tn._new_tensor_handle(t, scalar_kind)
        for idx in left_inds
            push!(left_handles, tn._new_index_handle(idx))
        end

        out_u = Ref{Ptr{Cvoid}}(C_NULL)
        out_s = Ref{Ptr{Cvoid}}(C_NULL)
        out_v = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            tn._t4a(:t4a_tensor_svd),
            Cint,
            (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t, Cdouble, Cdouble, Csize_t,
             Ref{Ptr{Cvoid}}, Ref{Ptr{Cvoid}}, Ref{Ptr{Cvoid}}),
            t_handle, left_handles, Csize_t(length(left_handles)),
            float(rtol), float(cutoff), Csize_t(maxdim),
            out_u, out_s, out_v,
        )
        tn._check_backend_status(status, "computing tensor SVD")
        u_handle = out_u[]
        s_handle = out_s[]
        v_handle = out_v[]
        return (tn._tensor_from_handle(u_handle),
                tn._tensor_from_handle(s_handle),
                tn._tensor_from_handle(v_handle))
    finally
        tn._release_tensor_handle(v_handle)
        tn._release_tensor_handle(s_handle)
        tn._release_tensor_handle(u_handle)
        for h in reverse(left_handles)
            tn._release_index_handle(h)
        end
        tn._release_tensor_handle(t_handle)
    end
end

LinearAlgebra.svd(t::Tensor, left_inds::Index...; kwargs...) = svd(t, collect(left_inds); kwargs...)
```

- [ ] **Step 4: Export svd**

Add `svd` to exports in `src/Tensor4all.jl`:

```julia
export svd
```

- [ ] **Step 5: Run tests**

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/Core/Tensor.jl src/Tensor4all.jl test/core/tensor_factorize.jl
git commit -m "feat: add Tensor SVD via t4a_tensor_svd C API"
```

---

## Task 4: Tensor QR via C API

**Files:**
- Modify: `src/Core/Tensor.jl`
- Modify: `test/core/tensor_factorize.jl`

- [ ] **Step 1: Write failing tests**

Add to `test/core/tensor_factorize.jl`:

```julia
@testset "Tensor QR" begin
    i = Index(3; tags=["i"])
    j = Index(4; tags=["j"])
    data = reshape(collect(1.0:12.0), 3, 4)
    t = Tensor(data, [i, j])

    @testset "basic QR" begin
        Q, R = qr(t, [i])
        reconstructed = contract(Q, R)
        @test isapprox(t, reconstructed; atol=1e-12)
    end

    @testset "vararg form" begin
        Q, R = qr(t, i)
        reconstructed = contract(Q, R)
        @test isapprox(t, reconstructed; atol=1e-12)
    end

    @testset "rank-3 tensor QR" begin
        k = Index(2; tags=["k"])
        data3 = reshape(collect(1.0:24.0), 3, 4, 2)
        t3 = Tensor(data3, [i, j, k])
        Q, R = qr(t3, [i, j])
        reconstructed = contract(Q, R)
        @test isapprox(t3, reconstructed; atol=1e-10)
    end
end
```

- [ ] **Step 2: Implement QR**

Add to `src/Core/Tensor.jl`:

```julia
import LinearAlgebra: qr

"""
    qr(t, left_inds::Vector{Index})

Compute QR factorization of tensor `t`, grouping `left_inds` on the left.
Returns `(Q, R)` as Tensors. Reconstruct via `contract(Q, R)`.
"""
function LinearAlgebra.qr(t::Tensor, left_inds::Vector{Index})
    isempty(left_inds) && throw(ArgumentError("left_inds must not be empty"))
    for idx in left_inds
        idx in inds(t) || throw(ArgumentError("Index $idx not found in tensor indices $(inds(t))"))
    end
    length(left_inds) == rank(t) && throw(ArgumentError("left_inds must not contain all indices"))

    scalar_kind = _tensor_scalar_kind(t)
    tn = _tensor_networks_module()
    t_handle = C_NULL
    left_handles = Ptr{Cvoid}[]
    q_handle = C_NULL
    r_handle = C_NULL

    try
        t_handle = tn._new_tensor_handle(t, scalar_kind)
        for idx in left_inds
            push!(left_handles, tn._new_index_handle(idx))
        end

        out_q = Ref{Ptr{Cvoid}}(C_NULL)
        out_r = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            tn._t4a(:t4a_tensor_qr),
            Cint,
            (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t, Ref{Ptr{Cvoid}}, Ref{Ptr{Cvoid}}),
            t_handle, left_handles, Csize_t(length(left_handles)),
            out_q, out_r,
        )
        tn._check_backend_status(status, "computing tensor QR")
        q_handle = out_q[]
        r_handle = out_r[]
        return (tn._tensor_from_handle(q_handle), tn._tensor_from_handle(r_handle))
    finally
        tn._release_tensor_handle(r_handle)
        tn._release_tensor_handle(q_handle)
        for h in reverse(left_handles)
            tn._release_index_handle(h)
        end
        tn._release_tensor_handle(t_handle)
    end
end

LinearAlgebra.qr(t::Tensor, left_inds::Index...) = qr(t, collect(left_inds))
```

- [ ] **Step 3: Export qr**

Add `qr` to exports in `src/Tensor4all.jl`.

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/Core/Tensor.jl src/Tensor4all.jl test/core/tensor_factorize.jl
git commit -m "feat: add Tensor QR via t4a_tensor_qr C API"
```

---

## Task 5: TensorTrain dag, linkinds, linkdims, siteinds

**Files:**
- Modify: `src/TensorNetworks/backend/treetn.jl`
- Modify: `src/TensorNetworks.jl`
- Modify: `src/Tensor4all.jl`
- Create: `test/tensornetworks/queries.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write failing tests**

Create `test/tensornetworks/queries.jl`:

```julia
using Test
using Tensor4all

@testset "TensorTrain queries" begin
    # Build a 3-site MPS: t1[s1,l1] — t2[l1,s2,l2] — t3[l2,s3]
    s1 = Index(2; tags=["s1"])
    s2 = Index(2; tags=["s2"])
    s3 = Index(2; tags=["s3"])
    l1 = Index(3; tags=["l1"])
    l2 = Index(3; tags=["l2"])

    t1 = Tensor(randn(2, 3), [s1, l1])
    t2 = Tensor(randn(3, 2, 3), [l1, s2, l2])
    t3 = Tensor(randn(3, 2), [l2, s3])
    tt = TensorNetworks.TensorTrain([t1, t2, t3])

    @testset "linkinds(tt)" begin
        links = TensorNetworks.linkinds(tt)
        @test length(links) == 2
        @test links[1] == l1
        @test links[2] == l2
    end

    @testset "linkinds(tt, i)" begin
        @test TensorNetworks.linkinds(tt, 1) == l1
        @test TensorNetworks.linkinds(tt, 2) == l2
    end

    @testset "linkdims(tt)" begin
        dims = TensorNetworks.linkdims(tt)
        @test dims == [3, 3]
    end

    @testset "siteinds(tt)" begin
        sites = TensorNetworks.siteinds(tt)
        @test length(sites) == 3
        @test s1 in sites[1]
        @test s2 in sites[2]
        @test s3 in sites[3]
    end

    @testset "siteinds(tt, i)" begin
        @test s1 in TensorNetworks.siteinds(tt, 1)
        @test s2 in TensorNetworks.siteinds(tt, 2)
    end

    @testset "1-site TT (no links)" begin
        tt1 = TensorNetworks.TensorTrain([Tensor(randn(2), [s1])])
        @test TensorNetworks.linkinds(tt1) == []
        @test length(TensorNetworks.siteinds(tt1)) == 1
    end

    @testset "dag" begin
        data_c = Complex{Float64}[1+2im, 3+4im, 5+6im, 7+8im, 9+10im, 11+12im]
        tc = Tensor(reshape(data_c, 2, 3), [s1, l1])
        tt_c = TensorNetworks.TensorTrain([tc])
        tt_d = TensorNetworks.dag(tt_c)
        @test tt_d[1].data ≈ conj(tc.data)
        @test length(tt_d) == 1
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `linkinds` not defined

- [ ] **Step 3: Implement linkinds, linkdims, siteinds, dag**

Add to `src/TensorNetworks/backend/treetn.jl` (or a new file `src/TensorNetworks/queries.jl` — choose based on file size):

```julia
"""
    linkinds(tt)

Return bond indices between adjacent tensors. Vector of length `length(tt) - 1`.
"""
function linkinds(tt::TensorTrain)
    isempty(tt.data) && return Index[]
    links = Index[]
    for i in 1:(length(tt) - 1)
        shared = commoninds(inds(tt[i]), inds(tt[i + 1]))
        length(shared) == 1 || throw(ArgumentError(
            "Expected exactly 1 shared index between sites $i and $(i+1), got $(length(shared))",
        ))
        push!(links, only(shared))
    end
    return links
end

"""
    linkinds(tt, i)

Return the bond index between site `i` and site `i+1`.
"""
function linkinds(tt::TensorTrain, i::Integer)
    shared = commoninds(inds(tt[i]), inds(tt[i + 1]))
    length(shared) == 1 || throw(ArgumentError(
        "Expected exactly 1 shared index between sites $i and $(i+1), got $(length(shared))",
    ))
    return only(shared)
end

"""
    linkdims(tt)

Return bond dimensions between adjacent tensors.
"""
linkdims(tt::TensorTrain) = [dim(idx) for idx in linkinds(tt)]

"""
    siteinds(tt)

Return site indices (non-bond) at each tensor. Vector of vectors.
"""
function siteinds(tt::TensorTrain)
    isempty(tt.data) && return Vector{Index}[]
    links = Set{Index}()
    for i in 1:(length(tt) - 1)
        for idx in commoninds(inds(tt[i]), inds(tt[i + 1]))
            push!(links, idx)
        end
    end
    return [filter(idx -> !(idx in links), inds(tt[i])) for i in 1:length(tt)]
end

"""
    siteinds(tt, i)

Return site indices (non-bond) at tensor `i`.
"""
function siteinds(tt::TensorTrain, i::Integer)
    bond_inds = Set{Index}()
    if i > 1
        for idx in commoninds(inds(tt[i]), inds(tt[i - 1]))
            push!(bond_inds, idx)
        end
    end
    if i < length(tt)
        for idx in commoninds(inds(tt[i]), inds(tt[i + 1]))
            push!(bond_inds, idx)
        end
    end
    return filter(idx -> !(idx in bond_inds), inds(tt[i]))
end

"""
    dag(tt)

Return a new TensorTrain with all site tensors conjugated.
"""
function dag(tt::TensorTrain)
    return TensorTrain([dag(t) for t in tt.data], tt.llim, tt.rlim)
end
```

- [ ] **Step 4: Export**

Add to `src/TensorNetworks.jl` exports:

```julia
export dag, linkinds, linkdims, siteinds
```

Add to `src/Tensor4all.jl`:

```julia
export linkinds, linkdims, siteinds
```

(Note: `dag` is already exported from Core)

- [ ] **Step 5: Add test include**

Add to `test/runtests.jl`:

```julia
include("tensornetworks/queries.jl")
```

- [ ] **Step 6: Run tests**

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/TensorNetworks/backend/treetn.jl src/TensorNetworks.jl src/Tensor4all.jl test/tensornetworks/queries.jl test/runtests.jl
git commit -m "feat: add TensorTrain dag, linkinds, linkdims, siteinds"
```

---

## Task 6: TensorTrain orthogonalize and truncate

**Prerequisite:** tensor4all-rs pin updated (for canonical_region sync in _treetn_from_handle).

**Files:**
- Modify: `src/TensorNetworks/backend/treetn.jl`
- Modify: `src/TensorNetworks.jl`
- Modify: `src/Tensor4all.jl`
- Create: `test/tensornetworks/orthogonalize_truncate.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write failing tests**

Create `test/tensornetworks/orthogonalize_truncate.jl`:

```julia
using Test
using Tensor4all

@testset "TensorTrain orthogonalize and truncate" begin
    s1 = Index(2; tags=["s1"])
    s2 = Index(2; tags=["s2"])
    s3 = Index(2; tags=["s3"])
    l1 = Index(4; tags=["l1"])
    l2 = Index(4; tags=["l2"])

    t1 = Tensor(randn(2, 4), [s1, l1])
    t2 = Tensor(randn(4, 2, 4), [l1, s2, l2])
    t3 = Tensor(randn(4, 2), [l2, s3])
    tt = TensorNetworks.TensorTrain([t1, t2, t3])

    @testset "orthogonalize" begin
        tt2 = TensorNetworks.orthogonalize(tt, 2)
        @test length(tt2) == 3
        # llim/rlim should reflect center at site 2
        @test tt2.llim == 1
        @test tt2.rlim == 3
        # Original unchanged
        @test tt.llim == 0
        @test tt.rlim == 4
    end

    @testset "orthogonalize preserves norm" begin
        n1 = TensorNetworks.norm(tt)
        tt2 = TensorNetworks.orthogonalize(tt, 1)
        n2 = TensorNetworks.norm(tt2)
        @test n1 ≈ n2 rtol=1e-10
    end

    @testset "truncate with maxdim" begin
        tt2 = TensorNetworks.truncate(tt; maxdim=2)
        dims = TensorNetworks.linkdims(tt2)
        @test all(d -> d <= 2, dims)
    end

    @testset "truncate preserves approximate norm" begin
        n1 = TensorNetworks.norm(tt)
        tt2 = TensorNetworks.truncate(tt; maxdim=3)
        n2 = TensorNetworks.norm(tt2)
        @test n2 <= n1 * 1.01  # truncation can only reduce or preserve norm
    end

    @testset "error: empty TT" begin
        empty_tt = TensorNetworks.TensorTrain(Tensor[])
        @test_throws ArgumentError TensorNetworks.orthogonalize(empty_tt, 1)
    end

    @testset "error: site out of range" begin
        @test_throws ArgumentError TensorNetworks.orthogonalize(tt, 0)
        @test_throws ArgumentError TensorNetworks.orthogonalize(tt, 4)
    end
end
```

- [ ] **Step 2: Implement orthogonalize and truncate**

Add to `src/TensorNetworks/backend/treetn.jl`:

```julia
const _T4A_CANONICAL_FORM_UNITARY = Cint(0)
const _T4A_CANONICAL_FORM_LU = Cint(1)

function _canonical_form_code(form::Symbol)
    if form === :unitary
        return _T4A_CANONICAL_FORM_UNITARY
    elseif form === :lu
        return _T4A_CANONICAL_FORM_LU
    end
    throw(ArgumentError("unknown canonical form $form. Expected :unitary or :lu"))
end

"""
    orthogonalize(tt, site; form=:unitary)

Return a new TensorTrain orthogonalized to the given site.
"""
function orthogonalize(tt::TensorTrain, site::Integer; form::Symbol=:unitary)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for orthogonalize"))
    1 <= site <= length(tt) || throw(ArgumentError(
        "site must be in 1:$(length(tt)), got $site",
    ))

    scalar_kind = _promoted_scalar_kind(tt)
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    try
        status = ccall(
            _t4a(:t4a_treetn_orthogonalize),
            Cint,
            (Ptr{Cvoid}, Csize_t, Cint),
            tt_handle,
            Csize_t(site - 1),
            _canonical_form_code(form),
        )
        _check_backend_status(status, "orthogonalizing TensorTrain to site $site")
        return _treetn_from_handle(tt_handle)
    finally
        _release_treetn_handle(tt_handle)
    end
end

"""
    truncate(tt; rtol=0.0, cutoff=0.0, maxdim=0)

Return a new TensorTrain with reduced bond dimensions.
"""
function truncate(tt::TensorTrain; rtol::Real=0.0, cutoff::Real=0.0, maxdim::Integer=0)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for truncate"))
    (rtol == 0.0 && cutoff == 0.0 && maxdim == 0) && throw(ArgumentError(
        "At least one of rtol, cutoff, or maxdim must be specified",
    ))

    scalar_kind = _promoted_scalar_kind(tt)
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    try
        status = ccall(
            _t4a(:t4a_treetn_truncate),
            Cint,
            (Ptr{Cvoid}, Cdouble, Cdouble, Csize_t),
            tt_handle,
            float(rtol),
            float(cutoff),
            Csize_t(maxdim),
        )
        _check_backend_status(status, "truncating TensorTrain")
        return _treetn_from_handle(tt_handle)
    finally
        _release_treetn_handle(tt_handle)
    end
end
```

- [ ] **Step 3: Export**

Add to `src/TensorNetworks.jl`:

```julia
export orthogonalize, truncate
```

Add to `src/Tensor4all.jl`:

```julia
export orthogonalize
# Note: truncate may conflict with Base.truncate — use TensorNetworks.truncate
```

- [ ] **Step 4: Add test include**

Add to `test/runtests.jl`:

```julia
include("tensornetworks/orthogonalize_truncate.jl")
```

- [ ] **Step 5: Run tests**

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/TensorNetworks/backend/treetn.jl src/TensorNetworks.jl src/Tensor4all.jl test/tensornetworks/orthogonalize_truncate.jl test/runtests.jl
git commit -m "feat: add TensorTrain orthogonalize and truncate via C API"
```

---

## Task 7: QuanticsTransform — Layout Handle and Shift Operator

**Files:**
- Modify: `src/QuanticsTransform.jl`
- Modify: `src/TensorNetworks/backend/capi.jl` (add QTT layout constants)
- Create: `test/quanticstransform/materialize.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write failing test**

Create `test/quanticstransform/materialize.jl`:

```julia
using Test
using Tensor4all
using LinearAlgebra: norm

@testset "QuanticsTransform materialization" begin
    @testset "shift_operator periodic" begin
        R = 3
        op = QuanticsTransform.shift_operator(R, 1; bc=:periodic)
        @test op.mpo !== nothing
        @test length(op.mpo) == R
        @test length(op.input_indices) == R
        @test length(op.output_indices) == R
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `op.mpo` is `nothing` (placeholder only)

- [ ] **Step 3: Add QTT layout C API bindings**

Add constants and helper to `src/TensorNetworks/backend/capi.jl` or a new file `src/QuanticsTransform/capi.jl`:

```julia
const _T4A_QTT_LAYOUT_FUSED = Cint(0)
const _T4A_QTT_LAYOUT_INTERLEAVED = Cint(1)
const _T4A_BC_PERIODIC = Cint(0)
const _T4A_BC_OPEN = Cint(1)

function _bc_code(bc::Symbol)
    bc === :periodic && return _T4A_BC_PERIODIC
    bc === :open && return _T4A_BC_OPEN
    throw(ArgumentError("unknown boundary condition $bc. Expected :periodic or :open"))
end

function _new_qtt_layout_handle(nvars::Integer, resolutions::Vector{<:Integer})
    out = Ref{Ptr{Cvoid}}(C_NULL)
    res_c = Csize_t[Csize_t(r) for r in resolutions]
    status = ccall(
        _t4a(:t4a_qtt_layout_new),
        Cint,
        (Cint, Csize_t, Ptr{Csize_t}, Ref{Ptr{Cvoid}}),
        _T4A_QTT_LAYOUT_FUSED,
        Csize_t(nvars),
        res_c,
        out,
    )
    _check_backend_status(status, "creating QTT layout")
    return out[]
end

function _release_qtt_layout_handle(ptr::Ptr{Cvoid})
    ptr == C_NULL && return
    ccall(_t4a(:t4a_qtt_layout_release), Cvoid, (Ptr{Cvoid},), ptr)
end
```

- [ ] **Step 4: Add MPO index extraction helper**

```julia
"""
    _extract_mpo_io_indices(tt::TensorTrain)

Extract input and output indices from a materialized MPO.
Convention: per-site non-bond indices are ordered [output, input]
following the Rust C API tensor layout.
"""
function _extract_mpo_io_indices(tt::TensorTrain)
    sites = siteinds(tt)
    input_indices = Index[]
    output_indices = Index[]
    for site_inds in sites
        length(site_inds) == 2 || throw(ArgumentError(
            "Expected 2 site indices per MPO tensor, got $(length(site_inds))",
        ))
        push!(output_indices, site_inds[1])
        push!(input_indices, site_inds[2])
    end
    return input_indices, output_indices
end
```

- [ ] **Step 5: Implement shift_operator materialization**

Replace in `src/QuanticsTransform.jl`:

```julia
function shift_operator(r::Integer, offset::Integer; bc=:periodic)
    layout_handle = TensorNetworks._new_qtt_layout_handle(1, [r])
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            TensorNetworks._t4a(:t4a_qtransform_shift_materialize),
            Cint,
            (Ptr{Cvoid}, Csize_t, Int64, Cint, Ref{Ptr{Cvoid}}),
            layout_handle,
            Csize_t(0),  # target_var
            Int64(offset),
            TensorNetworks._bc_code(bc),
            out,
        )
        TensorNetworks._check_backend_status(status, "materializing shift operator")
        tt = TensorNetworks._treetn_from_handle(out[])
        TensorNetworks._release_treetn_handle(out[])

        input_indices, output_indices = TensorNetworks._extract_mpo_io_indices(tt)
        return TensorNetworks.LinearOperator(;
            mpo=tt,
            input_indices=input_indices,
            output_indices=output_indices,
        )
    finally
        TensorNetworks._release_qtt_layout_handle(layout_handle)
    end
end
```

- [ ] **Step 6: Run tests**

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/QuanticsTransform.jl src/TensorNetworks/backend/capi.jl test/quanticstransform/materialize.jl test/runtests.jl
git commit -m "feat: materialize shift_operator via t4a_qtransform_shift_materialize"
```

---

## Task 8: QuanticsTransform — Remaining Single-Target Operators

**Files:**
- Modify: `src/QuanticsTransform.jl`
- Modify: `test/quanticstransform/materialize.jl`

- [ ] **Step 1: Implement flip, cumsum, phase_rotation, fourier operators**

Each follows the same pattern as shift. Replace each `_placeholder_operator` call. Example for flip:

```julia
function flip_operator(r::Integer; bc=:periodic)
    layout_handle = TensorNetworks._new_qtt_layout_handle(1, [r])
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            TensorNetworks._t4a(:t4a_qtransform_flip_materialize),
            Cint,
            (Ptr{Cvoid}, Csize_t, Cint, Ref{Ptr{Cvoid}}),
            layout_handle, Csize_t(0), TensorNetworks._bc_code(bc), out,
        )
        TensorNetworks._check_backend_status(status, "materializing flip operator")
        tt = TensorNetworks._treetn_from_handle(out[])
        TensorNetworks._release_treetn_handle(out[])
        input_indices, output_indices = TensorNetworks._extract_mpo_io_indices(tt)
        return TensorNetworks.LinearOperator(;
            mpo=tt, input_indices=input_indices, output_indices=output_indices,
        )
    finally
        TensorNetworks._release_qtt_layout_handle(layout_handle)
    end
end
```

Repeat for `cumsum_operator`, `phase_rotation_operator`, `fourier_operator`
with appropriate C API function and arguments. `fourier_operator` additionally
passes `forward`, `maxbonddim`, `tolerance`.

- [ ] **Step 2: Implement affine_operator**

`affine_operator` has a different signature — takes rational matrix/vector:

```julia
function affine_operator(r::Integer, a_num, a_den, b_num, b_den; bc=:periodic)
    layout_handle = TensorNetworks._new_qtt_layout_handle(1, [r])
    try
        a_num_c = Int64[Int64(a_num)]
        a_den_c = Int64[Int64(a_den)]
        b_num_c = Int64[Int64(b_num)]
        b_den_c = Int64[Int64(b_den)]
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            TensorNetworks._t4a(:t4a_qtransform_affine_materialize),
            Cint,
            (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
             Csize_t, Csize_t, Cint, Ref{Ptr{Cvoid}}),
            layout_handle, a_num_c, a_den_c, b_num_c, b_den_c,
            Csize_t(1), Csize_t(1),  # m=1, n=1 for univariate
            TensorNetworks._bc_code(bc), out,
        )
        TensorNetworks._check_backend_status(status, "materializing affine operator")
        tt = TensorNetworks._treetn_from_handle(out[])
        TensorNetworks._release_treetn_handle(out[])
        input_indices, output_indices = TensorNetworks._extract_mpo_io_indices(tt)
        return TensorNetworks.LinearOperator(;
            mpo=tt, input_indices=input_indices, output_indices=output_indices,
        )
    finally
        TensorNetworks._release_qtt_layout_handle(layout_handle)
    end
end
```

- [ ] **Step 3: Add comprehensive tests**

Add to `test/quanticstransform/materialize.jl`:

```julia
@testset "flip_operator" begin
    @testset "periodic" begin
        op = QuanticsTransform.flip_operator(3; bc=:periodic)
        @test op.mpo !== nothing
        @test length(op.mpo) == 3
    end
    @testset "open" begin
        op = QuanticsTransform.flip_operator(3; bc=:open)
        @test op.mpo !== nothing
    end
end

@testset "cumsum_operator" begin
    op = QuanticsTransform.cumsum_operator(3)
    @test op.mpo !== nothing
    @test length(op.mpo) == 3
end

@testset "phase_rotation_operator" begin
    for theta in [0.0, pi/4, pi/2, pi]
        op = QuanticsTransform.phase_rotation_operator(3, theta)
        @test op.mpo !== nothing
    end
end

@testset "fourier_operator" begin
    @testset "forward" begin
        op = QuanticsTransform.fourier_operator(3; forward=true)
        @test op.mpo !== nothing
    end
    @testset "inverse" begin
        op = QuanticsTransform.fourier_operator(3; forward=false)
        @test op.mpo !== nothing
    end
end

@testset "affine_operator" begin
    op = QuanticsTransform.affine_operator(3, 1, 1, 1, 2; bc=:periodic)
    @test op.mpo !== nothing
    @test length(op.mpo) == 3
end

@testset "shift_operator apply roundtrip" begin
    R = 2
    op = QuanticsTransform.shift_operator(R, 1; bc=:periodic)
    # Create a simple MPS state
    s_in = op.input_indices
    s_out = op.output_indices
    TensorNetworks.set_iospaces!(op, s_in, s_out)

    # Build a delta-function MPS at position 0
    l = Index(1; tags=["l"])
    t1 = Tensor(reshape([1.0, 0.0], 2, 1), [s_in[1], l])
    t2 = Tensor(reshape([1.0, 0.0], 1, 2), [l, s_in[2]])
    state = TensorNetworks.TensorTrain([t1, t2])

    result = TensorNetworks.apply(op, state)
    @test TensorNetworks.norm(result) > 0
end
```

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/QuanticsTransform.jl test/quanticstransform/materialize.jl
git commit -m "feat: materialize all single-target QuanticsTransform operators"
```

---

## Task 9: QuanticsTransform — Multivar Operators

**Files:**
- Modify: `src/QuanticsTransform.jl`
- Modify: `test/quanticstransform/materialize.jl`

- [ ] **Step 1: Implement multivar operators**

Each multivar operator uses the same C API with a `target_var` parameter and
a layout with `nvars` variables:

```julia
function shift_operator_multivar(r::Integer, offset::Integer, nvars::Integer, target::Integer; bc=:periodic)
    resolutions = fill(r, nvars)
    layout_handle = TensorNetworks._new_qtt_layout_handle(nvars, resolutions)
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            TensorNetworks._t4a(:t4a_qtransform_shift_materialize),
            Cint,
            (Ptr{Cvoid}, Csize_t, Int64, Cint, Ref{Ptr{Cvoid}}),
            layout_handle,
            Csize_t(target - 1),  # 0-based target_var
            Int64(offset),
            TensorNetworks._bc_code(bc),
            out,
        )
        TensorNetworks._check_backend_status(status, "materializing multivar shift operator")
        tt = TensorNetworks._treetn_from_handle(out[])
        TensorNetworks._release_treetn_handle(out[])
        input_indices, output_indices = TensorNetworks._extract_mpo_io_indices(tt)
        return TensorNetworks.LinearOperator(;
            mpo=tt, input_indices=input_indices, output_indices=output_indices,
        )
    finally
        TensorNetworks._release_qtt_layout_handle(layout_handle)
    end
end
```

Repeat pattern for `flip_operator_multivar` and `phase_rotation_operator_multivar`.

- [ ] **Step 2: Add multivar tests**

```julia
@testset "multivar operators" begin
    @testset "shift_operator_multivar" begin
        op = QuanticsTransform.shift_operator_multivar(2, 1, 2, 1; bc=:periodic)
        @test op.mpo !== nothing
        # 2 variables × 2 bits = 4 sites total (fused layout)
        @test length(op.mpo) > 0
    end

    @testset "flip_operator_multivar" begin
        op = QuanticsTransform.flip_operator_multivar(2, 2, 2; bc=:periodic)
        @test op.mpo !== nothing
    end

    @testset "phase_rotation_operator_multivar" begin
        op = QuanticsTransform.phase_rotation_operator_multivar(2, pi/4, 2, 1)
        @test op.mpo !== nothing
    end
end
```

- [ ] **Step 3: Run tests**

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/QuanticsTransform.jl test/quanticstransform/materialize.jl
git commit -m "feat: materialize multivar QuanticsTransform operators"
```

---

## Task 10: Dense Reference Tests for QuanticsTransform

**Files:**
- Modify: `test/quanticstransform/materialize.jl`

- [ ] **Step 1: Add dense reference verification tests**

```julia
@testset "Dense reference verification (R=2)" begin
    @testset "shift by +1 periodic" begin
        R = 2
        N = 2^R  # 4 states
        op = QuanticsTransform.shift_operator(R, 1; bc=:periodic)
        TensorNetworks.set_iospaces!(op, op.input_indices, op.output_indices)

        # Build each basis state, apply, check shift
        for idx in 0:(N-1)
            bits = digits(idx; base=2, pad=R)
            # Build MPS for this basis state
            tensors = Tensor[]
            links = Index[]
            for r in 1:R
                if r < R
                    l = Index(1; tags=["l$r"])
                    push!(links, l)
                end
                data = zeros(2)
                data[bits[r] + 1] = 1.0
                if r == 1
                    push!(tensors, Tensor(reshape(data, 2, 1), [op.input_indices[1], links[1]]))
                elseif r == R
                    push!(tensors, Tensor(reshape(data, 1, 2), [links[end], op.input_indices[R]]))
                else
                    push!(tensors, Tensor(reshape(data, 1, 2, 1), [links[r-1], op.input_indices[r], links[r]]))
                end
            end
            state = TensorNetworks.TensorTrain(tensors)
            result = TensorNetworks.apply(op, state)

            # Expected: shift by +1 mod N
            expected_idx = mod(idx + 1, N)
            expected_bits = digits(expected_idx; base=2, pad=R)

            # Verify result is the expected basis state
            @test TensorNetworks.norm(result) ≈ 1.0 atol=1e-10
        end
    end
end
```

- [ ] **Step 2: Run tests**

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test/quanticstransform/materialize.jl
git commit -m "test: dense reference verification for QuanticsTransform operators"
```

---

## Dependency Graph

```
Task 1 (Rust SVD+QR)        ─── tensor4all-rs PR
   ─── Rust PR must be merged, pin updated ───
Task 2 (dag + Array)         ─── no Rust dependency
Task 3 (Tensor SVD)          ─── depends on Task 1 (pin)
Task 4 (Tensor QR)           ─── depends on Task 1 (pin)
Task 5 (TT queries + dag)   ─── no Rust dependency
Task 6 (orthogonalize/trunc) ─── depends on pin (uses existing C API)
Task 7 (QT shift)            ─── no new Rust dependency
Task 8 (QT remaining)        ─── depends on Task 7 (shared infra)
Task 9 (QT multivar)         ─── depends on Task 8
Task 10 (dense tests)        ─── depends on Tasks 7-9
```

**Parallelizable:** Tasks 2, 5, 7 can start immediately (no new Rust dependency). Tasks 3, 4, 6 wait for Rust pin update.
