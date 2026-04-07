using Documenter
using Tensor4all

makedocs(
    sitename="Tensor4all.jl",
    modules=[Tensor4all],
    repo=Documenter.Remotes.GitHub("tensor4all", "Tensor4all.jl"),
    pages=[
        "Home" => "index.md",
        "Modules" => "modules.md",
        "API Reference" => [
            "Core (Index, Tensor)" => "api/core.md",
            "SimpleTT" => "api/simplett.md",
            "TreeTN (MPS/MPO)" => "api/treetn.md",
            "QuanticsGrids" => "api/quanticsgrids.md",
            "QuanticsTCI" => "api/quanticstci.md",
            "QuanticsTransform" => "api/quanticstransform.md",
            "TreeTCI" => "api/treetci.md",
        ],
        "Tutorials" => [
            "1D Quantics Interpolation" => "tutorials/quantics1d.md",
            "1D Fourier Transform" => "tutorials/fourier1d.md",
        ],
    ],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tensor4all.github.io/Tensor4all.jl",
    ),
    warnonly=[:missing_docs],
)

deploydocs(
    repo="github.com/tensor4all/Tensor4all.jl.git",
    devbranch="main",
    push_preview=true,
)
