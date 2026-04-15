using HDF5: HDF5, attributes, create_group, h5open, open_group, read, write

const _MPS_TYPE = "MPS"
const _INDEX_TYPE = "Index"
const _INDEXSET_TYPE = "IndexSet"
const _ITENSOR_TYPE = "ITensor"
const _TAGSET_TYPE = "TagSet"
const _VERSION = 1

function _write_tagset(
    parent::Union{HDF5.File,HDF5.Group},
    name::AbstractString,
    value::AbstractVector{<:AbstractString},
)
    g = create_group(parent, name)
    attributes(g)["type"] = _TAGSET_TYPE
    attributes(g)["version"] = _VERSION
    write(g, "tags", join(value, ","))
    return g
end

function _write_index(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, i::Index)
    g = create_group(parent, name)
    attributes(g)["type"] = _INDEX_TYPE
    attributes(g)["version"] = _VERSION
    attributes(g)["space_type"] = "Int"
    write(g, "id", id(i))
    write(g, "dim", dim(i))
    write(g, "dir", 1)
    _write_tagset(g, "tags", tags(i))
    write(g, "plev", plev(i))
    return g
end

function _write_indexset(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, value::Vector{Index})
    g = create_group(parent, name)
    attributes(g)["type"] = _INDEXSET_TYPE
    attributes(g)["version"] = _VERSION
    write(g, "length", length(value))
    for (n, ind) in enumerate(value)
        _write_index(g, "index_$n", ind)
    end
    return g
end

_dense_typestr(::Type{T}) where {T} = "Dense{$T}"

function _write_itensor(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, t::Tensor)
    g = create_group(parent, name)
    attributes(g)["type"] = _ITENSOR_TYPE
    attributes(g)["version"] = _VERSION
    data, tensor_inds = _dense_array(t)
    _write_indexset(g, "inds", tensor_inds)
    storage = create_group(g, "storage")
    attributes(storage)["type"] = _dense_typestr(eltype(data))
    attributes(storage)["version"] = _VERSION
    write(storage, "data", vec(data))
    return g
end

function _read_tagset(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString)
    g = open_group(parent, name)
    read(attributes(g)["type"]) == _TAGSET_TYPE || error("HDF5 group '$name' does not contain TagSet data")
    tagstring = read(g, "tags")
    isempty(tagstring) && return String[]
    return split(tagstring, ",")
end

function _read_index(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString)
    g = open_group(parent, name)
    read(attributes(g)["type"]) == _INDEX_TYPE || error("HDF5 group '$name' does not contain Index data")
    # Tensor4all.Index does not model ITensors arrow direction, so HDF5 loads
    # accept the field for compatibility and discard it.
    read(g, "dir")
    return Index(
        read(g, "dim");
        tags=_read_tagset(g, "tags"),
        plev=read(g, "plev"),
        id=UInt64(read(g, "id")),
    )
end

function _read_indexset(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString)
    g = open_group(parent, name)
    read(attributes(g)["type"]) == _INDEXSET_TYPE || error("HDF5 group '$name' does not contain IndexSet data")
    n = read(g, "length")
    return Index[_read_index(g, "index_$j") for j in 1:n]
end

function _read_itensor(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString)
    g = open_group(parent, name)
    read(attributes(g)["type"]) == _ITENSOR_TYPE || error("HDF5 group '$name' does not contain ITensor data")
    tensor_inds = _read_indexset(g, "inds")
    storage = open_group(g, "storage")
    typestr = read(attributes(storage)["type"])
    startswith(typestr, "Dense{") || error("Unsupported storage type '$typestr'")
    data = read(storage, "data")
    return Tensor(reshape(data, dim.(tensor_inds)...), tensor_inds)
end

"""
    save_as_mps(path, name, tt)

Write a `TensorTrain` using the ITensorMPS-compatible `MPS` HDF5 schema.
"""
function save_as_mps(path::AbstractString, name::AbstractString, tt::TensorTrain)
    h5open(path, "cw") do f
        g = create_group(f, name)
        attributes(g)["type"] = _MPS_TYPE
        attributes(g)["version"] = _VERSION
        write(g, "length", length(tt))
        write(g, "llim", tt.llim)
        write(g, "rlim", tt.rlim)
        for (n, tensor) in enumerate(tt.data)
            _write_itensor(g, "MPS[$n]", tensor)
        end
    end
    return path
end

"""
    load_tt(path, name)

Read an ITensorMPS-compatible `MPS` HDF5 group into a `TensorTrain`.
"""
function load_tt(path::AbstractString, name::AbstractString)
    return h5open(path, "r") do f
        g = open_group(f, name)
        read(attributes(g)["type"]) == _MPS_TYPE || error("HDF5 group or file does not contain MPS data")
        n = read(g, "length")
        llim = read(g, "llim")
        rlim = read(g, "rlim")
        tensors = [_read_itensor(g, "MPS[$i]") for i in 1:n]
        TensorTrain(tensors, llim, rlim)
    end
end
