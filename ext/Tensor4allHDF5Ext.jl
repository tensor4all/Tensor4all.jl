module Tensor4allHDF5Ext

using HDF5: HDF5, attributes, create_group, h5open, open_group, read, write
using Tensor4all
import Tensor4all.TensorNetworks: load_tt, save_as_mps

const _MPS_TYPE = "MPS"
const _INDEX_TYPE = "Index"
const _INDEXSET_TYPE = "IndexSet"
const _ITENSOR_TYPE = "ITensor"
const _TAGSET_TYPE = "TagSet"
const _VERSION = 1

function _write_tagset(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, tags::Vector{String})
    g = create_group(parent, name)
    attributes(g)["type"] = _TAGSET_TYPE
    attributes(g)["version"] = _VERSION
    write(g, "tags", join(tags, ","))
    return g
end

function _write_index(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, i::Tensor4all.Index)
    g = create_group(parent, name)
    attributes(g)["type"] = _INDEX_TYPE
    attributes(g)["version"] = _VERSION
    attributes(g)["space_type"] = "Int"
    write(g, "id", Tensor4all.id(i))
    write(g, "dim", Tensor4all.dim(i))
    write(g, "dir", 1)
    _write_tagset(g, "tags", Tensor4all.tags(i))
    write(g, "plev", Tensor4all.plev(i))
    return g
end

function _write_indexset(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, inds::Vector{Tensor4all.Index})
    g = create_group(parent, name)
    attributes(g)["type"] = _INDEXSET_TYPE
    attributes(g)["version"] = _VERSION
    write(g, "length", length(inds))
    for (n, ind) in enumerate(inds)
        _write_index(g, "index_$n", ind)
    end
    return g
end

function _dense_typestr(::Type{T}) where {T}
    return "Dense{$T}"
end

function _write_itensor(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, t::Tensor4all.Tensor)
    g = create_group(parent, name)
    attributes(g)["type"] = _ITENSOR_TYPE
    attributes(g)["version"] = _VERSION
    _write_indexset(g, "inds", Tensor4all.inds(t))
    storage = create_group(g, "storage")
    data = Tensor4all.copy_data(t)
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
    dir = read(g, "dir")
    dir == 1 || error("Cannot load Index with dir=$dir into Tensor4all.Index. Only dir=1 is supported.")
    return Tensor4all.Index(
        read(g, "dim");
        tags=_read_tagset(g, "tags"),
        plev=read(g, "plev"),
        id=Int(read(g, "id")),
    )
end

function _read_indexset(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString)
    g = open_group(parent, name)
    read(attributes(g)["type"]) == _INDEXSET_TYPE || error("HDF5 group '$name' does not contain IndexSet data")
    n = read(g, "length")
    return Tensor4all.Index[_read_index(g, "index_$j") for j in 1:n]
end

function _read_itensor(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString)
    g = open_group(parent, name)
    read(attributes(g)["type"]) == _ITENSOR_TYPE || error("HDF5 group '$name' does not contain ITensor data")
    inds = _read_indexset(g, "inds")
    storage = open_group(g, "storage")
    typestr = read(attributes(storage)["type"])
    startswith(typestr, "Dense{") || error("Unsupported storage type '$typestr'")
    data = read(storage, "data")
    return Tensor4all.Tensor(reshape(data, Tensor4all.dim.(inds)...), inds)
end

function save_as_mps(
    path::AbstractString,
    name::AbstractString,
    tt::Tensor4all.TensorNetworks.TensorTrain,
)
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

function load_tt(path::AbstractString, name::AbstractString)
    return h5open(path, "r") do f
        g = open_group(f, name)
        read(attributes(g)["type"]) == _MPS_TYPE || error("HDF5 group or file does not contain MPS data")
        n = read(g, "length")
        llim = read(g, "llim")
        rlim = read(g, "rlim")
        tensors = [_read_itensor(g, "MPS[$i]") for i in 1:n]
        Tensor4all.TensorNetworks.TensorTrain(tensors, llim, rlim)
    end
end

save_hdf5(path::AbstractString, name::AbstractString, tt::Tensor4all.TensorNetworks.TensorTrain) =
    Tensor4all.TensorNetworks.save_as_mps(path, name, tt)

load_hdf5(path::AbstractString, name::AbstractString) = Tensor4all.TensorNetworks.load_tt(path, name)

end # module
