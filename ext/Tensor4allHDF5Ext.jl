module Tensor4allHDF5Ext

using HDF5
using Tensor4all

save_hdf5(args...) = throw(Tensor4all.SkeletonNotImplemented(:save_hdf5, :extensions))
load_hdf5(args...) = throw(Tensor4all.SkeletonNotImplemented(:load_hdf5, :extensions))

end # module
