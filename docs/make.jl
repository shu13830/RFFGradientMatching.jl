using RFFGradientMatching
using Documenter

DocMeta.setdocmeta!(RFFGradientMatching, :DocTestSetup, :(using RFFGradientMatching); recursive=true)

makedocs(;
    modules=[RFFGradientMatching],
    authors="Anonymous",
    sitename="RFFGradientMatching.jl",
    format=Documenter.HTML(;
        canonical="https://anonymous-author.github.io/RFFGradientMatching.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/anonymous-author/RFFGradientMatching.jl",
    devbranch="main",
)
