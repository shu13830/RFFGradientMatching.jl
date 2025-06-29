using RFFGradientMatching
using Documenter

DocMeta.setdocmeta!(RFFGradientMatching, :DocTestSetup, :(using RFFGradientMatching); recursive=true)

makedocs(;
    modules=[RFFGradientMatching],
    authors="Shuichi Miyazawa",
    sitename="RFFGradientMatching.jl",
    format=Documenter.HTML(;
        canonical="https://shu13830.github.io/RFFGradientMatching.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/shu13830/RFFGradientMatching.jl",
    devbranch="main",
)
