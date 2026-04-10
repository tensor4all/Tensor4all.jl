using Documenter
using Tensor4all

makedocs(
    sitename="Tensor4all.jl",
    modules=[Tensor4all],
    repo=Documenter.Remotes.GitHub("tensor4all", "Tensor4all.jl"),
    pages=[
        "Home" => "index.md",
        "Architecture Status" => "modules.md",
        "Design Documents" => "design_documents.md",
        "Deferred Rework Plan" => "deferred_rework_plan.md",
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
