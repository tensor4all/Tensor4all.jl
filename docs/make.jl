using Documenter
using Tensor4all

makedocs(
    sitename="Tensor4all.jl",
    modules=[Tensor4all],
    repo=Documenter.Remotes.GitHub("tensor4all", "Tensor4all.jl"),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Architecture Status" => "modules.md",
        "API Reference" => "api.md",
        "Truncation Policy" => "truncation_policy.md",
        "Design Documents" => "design_documents.md",
        "Deferred Rework Plan" => "deferred_rework_plan.md",
    ],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tensor4all.github.io/Tensor4all.jl",
        size_threshold_warn=150 * 2^10,
    ),
    checkdocs=:exports,
    warnonly=[:missing_docs],
)
