"""
    TreeTCI

Tree-structured tensor cross interpolation via tensor4all-rs.

Provides `TreeTciGraph` for defining tree topologies and `SimpleTreeTci`
for running TCI on arbitrary tree structures. Results are returned as
`TreeTN.TreeTensorNetwork`.

# Usage
```julia
using Tensor4all.TreeTCI

graph = TreeTciGraph(4, [(0, 1), (1, 2), (2, 3)])
f(batch) = [sum(Float64, batch[:, j]) for j in 1:size(batch, 2)]
ttn, ranks, errors = crossinterpolate_tree(f, [3, 3, 3, 3], graph)
```
"""
module TreeTCI

using ..C_API
import ..TreeTN: TreeTensorNetwork

export TreeTciGraph, SimpleTreeTci
export crossinterpolate_tree

end # module TreeTCI
