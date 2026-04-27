# -----------------------------------------------------------------
# PGM 2026 — Figure and Table Generation
#
# Usage:
#   julia --project=. scripts/plot_figures.jl [--output_dir figures/]
#
# Reads CSV results from scripts/results/exp{N}/ and
# baselines/magi/results/ to generate all paper figures and table summaries.
# -----------------------------------------------------------------

using CSV, DataFrames, Statistics, Printf, ArgParse
using LaTeXStrings
using MCMCDiagnosticTools
using Plots; gr()

# ── Paths ────────────────────────────────────────────────────────
const RESULTS_BASE = joinpath(@__DIR__, "results")
const MAGI_RESULTS = joinpath(@__DIR__, "..", "baselines", "magi", "results")

# ── Style ────────────────────────────────────────────────────────
# PGM/PMLR: 1-column width ≈ 6.75 inch
const FIG_W = (487, 300)   # (width, height) in pixels at 72 dpi

const COLOR_GPGM  = RGB(0.275, 0.510, 0.706)  # steelblue
const COLOR_MAGI  = RGB(0.133, 0.545, 0.133)  # forestgreen
const COLOR_RFFGM = RGB(0.863, 0.078, 0.235)  # crimson

const METHOD_COLORS = Dict(
    "GPGM"  => COLOR_GPGM,
    "MAGI"  => COLOR_MAGI,
    "RFFGM" => COLOR_RFFGM,
)
const METHOD_MARKERS = Dict(
    "GPGM"  => :circle,
    "MAGI"  => :rect,
    "RFFGM" => :utriangle,
)
const METHOD_ORDER = ["GPGM", "MAGI", "RFFGM"]

method_color(m) = get(METHOD_COLORS, m, RGB(0.5, 0.5, 0.5))
method_marker(m) = get(METHOD_MARKERS, m, :circle)

default(fontfamily="Computer Modern", guidefontsize=10, tickfontsize=8,
    legendfontsize=8, titlefontsize=11, linewidth=1.5, grid=false,
    framestyle=:box, dpi=300)

# ── Helpers ──────────────────────────────────────────────────────

function load_csv(path::String)
    if !isfile(path)
        @warn "File not found: $path"
        return nothing
    end
    CSV.read(path, DataFrame)
end

expdir(n::Int) = joinpath(RESULTS_BASE, "exp$n")

function mean_se(x)
    m = mean(x)
    se = length(x) > 1 ? std(x) / sqrt(length(x)) : 0.0
    return m, se
end

function fmt_mean_se(m, se; digits=3)
    d = digits
    fmt = Printf.Format("%.$(d)f ± %.$(d)f")
    Printf.format(fmt, m, se)
end

# ── Fig.1: MCMC trajectory in (x(t_i), x(t_{i+1})) plane ────────
# Visualises how efficiently RFFGM walks through the x-x joint posterior
# compared with GPGM (where GP prior smoothness creates strong adjacent-state
# dependency). Data produced by run_exp1_state_scatter.jl (LV, N=50).

function plot_fig1_state_scatter(output_dir::String;
        seed::Int=42, dim::Int=1, pair_i::Int=7, traj_n::Int=300)
    @info "Generating Fig.1 (state-space MCMC trajectory)"
    dir = joinpath(RESULTS_BASE, "exp1_scatter")
    if !isdir(dir)
        @warn "Directory $dir not found — skipping Fig.1"
        return
    end
    meta_f = joinpath(dir, "exp1s_meta.csv")
    run_f  = joinpath(dir, "exp1s_run.csv")
    (!isfile(meta_f) || !isfile(run_f)) && (@warn "missing meta/run CSV"; return)
    df_meta = CSV.read(meta_f, DataFrame)
    df_run  = CSV.read(run_f, DataFrame)
    N      = df_run.N[1]
    t_peak = df_run.t_peak_idx[1]
    # Default: pair centred at (t_peak, t_peak+1)
    i = pair_i == 0 ? clamp(t_peak, 1, N-1) : pair_i
    j = i + 1
    t_i = df_meta.t[i]; t_j = df_meta.t[j]
    clean_ref = dim == 1 ? df_meta.x1_clean : df_meta.x2_clean

    function load_method(method)
        f = joinpath(dir, "exp1s_$(method)_seed$(seed)_chain.csv")
        isfile(f) || return nothing
        df = CSV.read(f, DataFrame)
        sub = sort!(filter(r -> r.dim == dim, df), :iter)
        return sub
    end

    dfs = Dict(m => load_method(m) for m in ("GPGM", "RFFGM"))
    (dfs["GPGM"] === nothing || dfs["RFFGM"] === nothing) && return

    # Global axis bounds from post-warmup of both methods
    xlo = Inf; xhi = -Inf; ylo = Inf; yhi = -Inf
    for m in ("GPGM", "RFFGM")
        sub = dfs[m]
        post = filter(r -> r.is_warmup == 0, sub)
        xi = post[!, Symbol("x_t$i")]; xj = post[!, Symbol("x_t$j")]
        xlo = min(xlo, minimum(xi)); xhi = max(xhi, maximum(xi))
        ylo = min(ylo, minimum(xj)); yhi = max(yhi, maximum(xj))
    end
    xpad = (xhi - xlo) * 0.05; ypad = (yhi - ylo) * 0.05
    xlim = (xlo - xpad, xhi + xpad); ylim = (ylo - ypad, yhi + ypad)

    plts = Plots.Plot[]
    panel_order = ("RFFGM", "GPGM")
    panel_titles = Dict("RFFGM" => "(a) RFFGM (Ours)", "GPGM" => "(b) GPGM")
    tl_global = 0
    ess_safe(x) = try
        Float64(MCMCDiagnosticTools.ess(reshape(x, :, 1, 1))[1])
    catch
        NaN
    end
    for (pidx, method) in enumerate(panel_order)
        sub = dfs[method]
        post = filter(r -> r.is_warmup == 0, sub)
        xi = post[!, Symbol("x_t$i")]; xj = post[!, Symbol("x_t$j")]

        step = mean(sqrt.(diff(xi).^2 .+ diff(xj).^2))
        ess_val = min(ess_safe(xi), ess_safe(xj))

        xlab = latexstring(@sprintf("x_{%d}(t{=}%.2f)", dim, t_i))
        ylab = pidx == 1 ? latexstring(@sprintf("x_{%d}(t{=}%.2f)", dim, t_j)) : ""
        p = plot(; xlabel=xlab, ylabel=ylab, title=panel_titles[method],
            xlims=xlim, ylims=ylim, legend=false, colorbar=false)

        # Background: post-warmup posterior density (slightly darker for visibility)
        scatter!(p, xi, xj;
            color=RGB(0.78, 0.78, 0.78), markersize=1.3, markerstrokewidth=0,
            alpha=0.55, label="")

        # Trajectory: consecutive post-warmup samples — dots only, gradient
        tl = min(traj_n, length(xi))
        tl_global = tl
        iters = collect(1:tl)
        scatter!(p, xi[1:tl], xj[1:tl];
            zcolor=iters, c=:viridis, clims=(1, tl), colorbar=false,
            markersize=2.2, markerstrokewidth=0, alpha=0.95, label="")

        # True value (black star)
        scatter!(p, [clean_ref[i]], [clean_ref[j]];
            marker=:star5, markersize=7, color=:black,
            markerstrokecolor=:black, markerstrokewidth=1, label="")

        annotate!(p, xlim[1] + xpad*2, ylim[2] - ypad*2,
            text(@sprintf("step = %.3f   ESS = %.0f", step, ess_val), 8, :left))
        push!(plts, p)
    end

    # Shared horizontal colourbar below the two equal-sized panels
    cb = heatmap(reshape(1:tl_global, 1, :);
        c=:viridis, clims=(1, tl_global),
        yaxis=false, yticks=false,
        xticks=([1, div(tl_global, 4), div(tl_global, 2), 3*div(tl_global, 4), tl_global],
                ["1", "$(div(tl_global,4))", "$(div(tl_global,2))",
                 "$(3*div(tl_global,4))", "$tl_global"]),
        xlabel="iteration", colorbar=false, framestyle=:box)

    l = @layout [grid(1, 2); a{0.12h}]
    fig = plot(plts[1], plts[2], cb; layout=l,
        size=(FIG_W[1], round(Int, FIG_W[2] * 0.85)),
        bottom_margin=1Plots.mm)
    path = joinpath(output_dir, "fig1_state_scatter.pdf")
    savefig(fig, path)
    @info "Saved $path  (dim=$dim pair=($i,$j) t=($(t_i),$(t_j)))"
end

# ── Fig.1 (legacy): MCMC Trajectory in (x₁(t_peak), θ_a) plane ──

function plot_fig1_trajectory(output_dir::String; seed::Int=42)
    @info "Generating Fig.1 (MCMC trajectory)"
    dir = joinpath(RESULTS_BASE, "exp1")
    if !isdir(dir)
        @warn "Directory $dir not found — skipping Fig.1 trajectory"
        return
    end

    # Load true values
    true_file = joinpath(dir, "exp1_true_values_seed$(seed).csv")
    df_true = load_csv(true_file)
    if df_true === nothing; return; end
    x_true = df_true.x1_tpeak_true[1]
    θ_true = df_true.theta_a_true[1]

    plts = Plots.Plot[]
    xlims_global = (Inf, -Inf)
    ylims_global = (Inf, -Inf)

    # First pass: compute shared axis limits
    for method in ["GPGM", "RFFGM"]
        bg_file = joinpath(dir, "exp1_background_$(method)_seed$(seed).csv")
        df_bg = load_csv(bg_file)
        if df_bg === nothing; continue; end
        xmin, xmax = extrema(df_bg.x1_tpeak)
        ymin, ymax = extrema(df_bg.theta_a)
        xlims_global = (min(xlims_global[1], xmin), max(xlims_global[2], xmax))
        ylims_global = (min(ylims_global[1], ymin), max(ylims_global[2], ymax))
    end
    # Add margin
    xpad = (xlims_global[2] - xlims_global[1]) * 0.05
    ypad = (ylims_global[2] - ylims_global[1]) * 0.05
    xlims_global = (xlims_global[1] - xpad, xlims_global[2] + xpad)
    ylims_global = (ylims_global[1] - ypad, ylims_global[2] + ypad)

    for (panel_idx, method) in enumerate(["GPGM", "RFFGM"])
        bg_file = joinpath(dir, "exp1_background_$(method)_seed$(seed).csv")
        traj_file = joinpath(dir, "exp1_trajectory_$(method)_seed$(seed).csv")
        df_bg = load_csv(bg_file)
        df_traj = load_csv(traj_file)
        if df_bg === nothing || df_traj === nothing; continue; end

        c = method_color(method)

        p = plot(; xlabel="x₁(t_peak)", ylabel=panel_idx == 1 ? "θ_a" : "",
            title="($('a' + panel_idx - 1)) $method",
            xlims=xlims_global, ylims=ylims_global, aspect_ratio=:auto)

        # Background: all samples (light grey)
        scatter!(p, df_bg.x1_tpeak, df_bg.theta_a;
            color=:lightgray, markersize=1, markerstrokewidth=0,
            alpha=0.3, label="")

        # Trajectory: consecutive samples with colored line
        plot!(p, df_traj.x1_tpeak, df_traj.theta_a;
            color=c, linewidth=1.5, alpha=0.8, label="")
        scatter!(p, df_traj.x1_tpeak, df_traj.theta_a;
            color=c, markersize=2.5, markerstrokewidth=0, alpha=0.8, label="")

        # Start marker (square)
        scatter!(p, [df_traj.x1_tpeak[1]], [df_traj.theta_a[1]];
            marker=:rect, markersize=6, color=c, markerstrokecolor=:black,
            markerstrokewidth=1.5, label="Start")

        # End marker (diamond)
        scatter!(p, [df_traj.x1_tpeak[end]], [df_traj.theta_a[end]];
            marker=:diamond, markersize=6, color=c, markerstrokecolor=:black,
            markerstrokewidth=1.5, label="End")

        # True value (star)
        scatter!(p, [x_true], [θ_true];
            marker=:star5, markersize=10, color=:gold, markerstrokecolor=:black,
            markerstrokewidth=1.5, label="True")

        # Mean step size annotation
        steps = [sqrt((df_traj.x1_tpeak[i+1]-df_traj.x1_tpeak[i])^2 +
                       (df_traj.theta_a[i+1]-df_traj.theta_a[i])^2)
                 for i in 1:nrow(df_traj)-1]
        ms = @sprintf("%.3f", mean(steps))
        annotate!(p, xlims_global[2] - xpad*2, ylims_global[1] + ypad*2,
            text("step = $ms", 7, :right))

        push!(plts, p)
    end

    fig = plot(plts...; layout=(1, 2), size=(FIG_W[1], round(Int, FIG_W[2] * 1.0)))
    path = joinpath(output_dir, "mcmc_trajectory.pdf")
    savefig(fig, path)
    @info "Saved $path"
end

# ── Fig.1 (old): Convergence Comparison (Exp1) ──────────────────

function plot_fig1(output_dir::String)
    @info "Generating Fig.1 (Exp1: Convergence)"
    dir = expdir(1)
    if !isdir(dir)
        @warn "Directory $dir not found — skipping Fig.1"
        return
    end

    files = filter(f -> endswith(f, "_logdens.csv"), readdir(dir; join=true))
    if isempty(files)
        @warn "No logdens CSVs found in $dir — skipping Fig.1"
        return
    end

    Ns = [10, 25, 40]
    plts = Plots.Plot[]

    for (panel_idx, N) in enumerate(Ns)
        p = plot(; xlabel="Iteration",
            ylabel=panel_idx == 1 ? "Log posterior density" : "",
            title="N = $N", legend=panel_idx == length(Ns) ? :bottomright : false)

        for method in ["GPGM", "RFFGM"]
            pattern = Regex("$(method)_N$(N)_seed\\d+_logdens\\.csv")
            matched = filter(f -> occursin(pattern, basename(f)), files)
            if isempty(matched); continue; end

            all_data = [CSV.read(f, DataFrame) for f in matched]
            iters = all_data[1].iteration
            vals = hcat([d.logdens for d in all_data]...)
            med = [median(vals[i, :]) for i in axes(vals, 1)]
            q25 = [quantile(vals[i, :], 0.25) for i in axes(vals, 1)]
            q75 = [quantile(vals[i, :], 0.75) for i in axes(vals, 1)]

            c = method_color(method)
            plot!(p, iters, med; ribbon=(med .- q25, q75 .- med),
                fillalpha=0.2, color=c, label=method)
        end
        push!(plts, p)
    end

    fig = plot(plts...; layout=(1, length(Ns)), size=(FIG_W[1], FIG_W[2]))
    path = joinpath(output_dir, "fig1_convergence.pdf")
    savefig(fig, path)
    @info "Saved $path"
end

# ── Fig.2: High-Dim Scaling (Exp4) ──────────────────────────────

function plot_fig2(output_dir::String)
    @info "Generating Fig.2 (Exp4: High-dim scaling)"

    df_julia = load_csv(joinpath(expdir(4), "exp4_summary.csv"))
    df_magi = load_csv(joinpath(MAGI_RESULTS, "lvc_summary.csv"))

    if df_julia === nothing && df_magi === nothing
        @warn "No Exp4 data found — skipping Fig.2"
        return
    end

    # Combine into unified DataFrame
    combined_rows = DataFrame[]
    if df_julia !== nothing
        for r in eachrow(df_julia)
            push!(combined_rows, DataFrame(method=r.method, K=r.K,
                ess_mean=r.ess_mean, rhat_max=r.rhat_max, seed=r.seed))
        end
    end
    if df_magi !== nothing
        for r in eachrow(df_magi)
            rhat = hasproperty(df_magi, :rhat_max) ? r.rhat_max : NaN
            push!(combined_rows, DataFrame(method="MAGI", K=r.K,
                ess_mean=r.ess_mean, rhat_max=rhat, seed=r.seed))
        end
    end
    df = vcat(combined_rows...)

    p1 = plot(; xlabel="K", ylabel="Mean ESS", title="Effective Sample Size",
        yscale=:log10, legend=:topright)
    p2 = plot(; xlabel="K", ylabel="Max R̂", title="Convergence Diagnostic",
        legend=:topleft)
    hline!(p2, [1.1]; color=:gray, linestyle=:dash, linewidth=1, label="R̂ = 1.1")

    for method in METHOD_ORDER
        sub = filter(r -> r.method == method, df)
        if nrow(sub) == 0; continue; end
        gdf = combine(groupby(sub, :K),
            :ess_mean => mean => :ess_avg,
            :rhat_max => (x -> mean(filter(!isnan, x))) => :rhat_avg)
        sort!(gdf, :K)
        c = method_color(method)
        mk = method_marker(method)

        plot!(p1, gdf.K, gdf.ess_avg; color=c, marker=mk, markersize=5, label=method)
        plot!(p2, gdf.K, gdf.rhat_avg; color=c, marker=mk, markersize=5, label=method)

        # Mark non-converged with ×
        bad = filter(r -> r.rhat_avg > 1.1, gdf)
        if nrow(bad) > 0
            scatter!(p2, bad.K, bad.rhat_avg;
                marker=:xcross, markersize=8, color=c, label="")
        end
    end

    fig = plot(p1, p2; layout=(1, 2), size=(FIG_W[1], round(Int, FIG_W[2] * 0.8)))
    path = joinpath(output_dir, "fig2_scaling.pdf")
    savefig(fig, path)
    @info "Saved $path"
end

# ── App.Fig.1: Trajectory Recovery (Exp6) ────────────────────────

function plot_appfig1(output_dir::String)
    @info "Generating App.Fig.1 (Exp6: Trajectory recovery)"
    dir = expdir(6)
    if !isdir(dir)
        @warn "Directory $dir not found — skipping App.Fig.1"
        return
    end

    obs_files = filter(f -> occursin("observations", f), readdir(dir; join=true))
    if isempty(obs_files)
        @warn "No observation CSVs in $dir — skipping App.Fig.1"
        return
    end
    df_obs = CSV.read(first(obs_files), DataFrame)

    traj_files = filter(f -> occursin("trajectory", f) && endswith(f, ".csv"),
        readdir(dir; join=true))
    if isempty(traj_files)
        @warn "No trajectory CSVs in $dir — skipping App.Fig.1"
        return
    end

    # Detect components from first trajectory file
    df_sample = CSV.read(first(traj_files), DataFrame)
    components = [replace(string(c), "mean_" => "")
        for c in filter(c -> startswith(string(c), "mean_"), names(df_sample))]

    plts = Plots.Plot[]
    for (k, comp) in enumerate(components)
        p = plot(; xlabel=k == length(components) ? "Time" : "",
            ylabel=uppercasefirst(comp), legend=k == 1 ? :topright : false)

        # Observations
        obs_col = Symbol("obs_$comp")
        clean_col = Symbol("clean_$comp")
        if hasproperty(df_obs, obs_col)
            scatter!(p, df_obs.t, df_obs[!, obs_col];
                color=:black, markersize=3, markerstrokewidth=0, label="Obs")
        end
        if hasproperty(df_obs, clean_col)
            plot!(p, df_obs.t, df_obs[!, clean_col];
                color=:black, linestyle=:dash, linewidth=1, label="True")
        end

        # Trajectories
        for f in sort(traj_files)
            df_t = CSV.read(f, DataFrame)
            fname = basename(f)
            method = if occursin("RFFGM", fname) "RFFGM"
            elseif occursin("GPGM", fname) "GPGM"
            elseif occursin("MAGI", fname) "MAGI"
            else continue end

            mean_col = Symbol("mean_$comp")
            lower_col = Symbol("lower_$comp")
            upper_col = Symbol("upper_$comp")
            if !hasproperty(df_t, mean_col); continue; end

            c = method_color(method)
            plot!(p, df_t.t, df_t[!, mean_col];
                ribbon=(df_t[!, mean_col] .- df_t[!, lower_col],
                        df_t[!, upper_col] .- df_t[!, mean_col]),
                fillalpha=0.15, color=c, label=method)
        end
        push!(plts, p)
    end

    fig = plot(plts...; layout=(length(components), 1),
        size=(FIG_W[1], FIG_W[2] * length(components) ÷ 2))
    path = joinpath(output_dir, "appfig1_trajectory.pdf")
    savefig(fig, path)
    @info "Saved $path"
end

# ── App.Fig.2: Lynx-Hare Real Data (Exp8) ───────────────────────

function plot_appfig2(output_dir::String)
    @info "Generating App.Fig.2 (Exp8: Lynx-Hare)"
    dir = expdir(8)

    df_data = load_csv(joinpath(dir, "exp8_data.csv"))
    if df_data === nothing
        @warn "No exp8_data.csv — skipping App.Fig.2"
        return
    end

    traj_files = filter(f -> occursin("trajectory", f) && endswith(f, ".csv"),
        readdir(dir; join=true))
    if isempty(traj_files)
        @warn "No trajectory CSVs in $dir — skipping App.Fig.2"
        return
    end

    components = ["hare", "lynx"]
    plts = Plots.Plot[]

    for (k, comp) in enumerate(components)
        p = plot(; xlabel=k == 2 ? "Time (normalized)" : "",
            ylabel=uppercasefirst(comp) * " (×1000)",
            legend=k == 1 ? :topright : false)

        data_col = Symbol(comp)
        if hasproperty(df_data, data_col)
            scatter!(p, df_data.t, df_data[!, data_col];
                color=:black, markersize=3, markerstrokewidth=0, label="Data")
        end

        for f in sort(traj_files)
            df_t = CSV.read(f, DataFrame)
            fname = basename(f)
            m = match(r"trajectory_(\w+?)_(\w+)\.csv", fname)
            if m === nothing; continue; end
            method, kernel = m.captures

            mean_col = Symbol("mean_$comp")
            lower_col = Symbol("lower_$comp")
            upper_col = Symbol("upper_$comp")
            if !hasproperty(df_t, mean_col); continue; end

            c = method_color(method)
            ls = kernel == "RBF" ? :solid : :dash
            plot!(p, df_t.t, df_t[!, mean_col];
                ribbon=(df_t[!, mean_col] .- df_t[!, lower_col],
                        df_t[!, upper_col] .- df_t[!, mean_col]),
                fillalpha=0.1, color=c, linestyle=ls, label="$method-$kernel")
        end
        push!(plts, p)
    end

    fig = plot(plts...; layout=(2, 1), size=(FIG_W[1], FIG_W[2]))
    path = joinpath(output_dir, "appfig2_lynxhare.pdf")
    savefig(fig, path)
    @info "Saved $path"
end

# ── App.Fig.3: RFF Ablation L sweep (Exp9) ──────────────────────

function plot_appfig3(output_dir::String)
    @info "Generating App.Fig.3 (Exp9: RFF ablation)"
    df = load_csv(joinpath(expdir(9), "exp9_summary.csv"))
    if df === nothing
        @warn "No exp9_summary.csv — skipping App.Fig.3"
        return
    end

    gdf = combine(groupby(df, :L),
        :rmsd => mean => :rmsd_mean,
        :rmsd => (x -> std(x)/sqrt(length(x))) => :rmsd_se,
        :rff_error => mean => :err_mean,
        :rff_error => (x -> std(x)/sqrt(length(x))) => :err_se)
    sort!(gdf, :L)

    p1 = plot(gdf.L, gdf.rmsd_mean; yerror=gdf.rmsd_se,
        color=COLOR_RFFGM, marker=:utriangle, markersize=5,
        xlabel="L (number of RFF features)", ylabel="RMSD",
        title="Parameter RMSD", legend=false)

    p2 = plot(gdf.L, gdf.err_mean; yerror=gdf.err_se,
        color=COLOR_RFFGM, marker=:utriangle, markersize=5,
        xlabel="L (number of RFF features)", ylabel="RFF error",
        title="Kernel Approximation Error", legend=false)

    fig = plot(p1, p2; layout=(1, 2), size=(FIG_W[1], round(Int, FIG_W[2] * 0.8)))
    path = joinpath(output_dir, "appfig3_rff_ablation.pdf")
    savefig(fig, path)
    @info "Saved $path"
end

# ── App.G: γ Sensitivity (Exp-γ) ─────────────────────────────────

function plot_gamma_sensitivity(output_dir::String)
    @info "Generating App.G (Exp-γ: γ sensitivity)"
    df = load_csv(joinpath(RESULTS_BASE, "exp_gamma", "exp_gamma_summary.csv"))
    if df === nothing
        @warn "No exp_gamma_summary.csv — skipping App.G"
        return
    end

    # Aggregate by method × γ
    gdf = combine(groupby(df, [:method, :gamma]),
        :rmsd => mean => :rmsd_mean,
        :rmsd => (x -> std(x)/sqrt(length(x))) => :rmsd_se,
        :ess_mean => (x -> mean(filter(isfinite, x))) => :ess_avg,
        :ess_mean => (x -> let v=filter(isfinite, x); length(v)>1 ? std(v)/sqrt(length(v)) : 0.0 end) => :ess_se,
        :rhat_max => (x -> mean(filter(isfinite, x))) => :rhat_avg)
    sort!(gdf, [:method, :gamma])

    p1 = plot(; xlabel="γ", ylabel="RMSD", title="Parameter Accuracy",
        xscale=:log10, legend=:topright)
    p2 = plot(; xlabel="γ", ylabel="Mean ESS", title="Sampling Efficiency",
        xscale=:log10, yscale=:log10, legend=:bottomright)
    p3 = plot(; xlabel="γ", ylabel="Max R̂", title="Convergence",
        xscale=:log10, legend=:topright)
    hline!(p3, [1.1]; color=:gray, linestyle=:dash, linewidth=1, label="R̂ = 1.1")

    for method in ["GPGM", "RFFGM"]
        sub = filter(r -> r.method == method, gdf)
        if nrow(sub) == 0; continue; end
        c = method_color(method)
        mk = method_marker(method)

        plot!(p1, sub.gamma, sub.rmsd_mean; yerror=sub.rmsd_se,
            color=c, marker=mk, markersize=5, label=method)
        # For ESS, replace NaN with a small value for plotting
        ess_plot = replace(sub.ess_avg, NaN => 1.0)
        plot!(p2, sub.gamma, ess_plot;
            color=c, marker=mk, markersize=5, label=method)
        rhat_plot = replace(sub.rhat_avg, NaN => 3.0)
        plot!(p3, sub.gamma, rhat_plot;
            color=c, marker=mk, markersize=5, label=method)
    end

    fig = plot(p1, p2, p3; layout=(1, 3), size=(round(Int, FIG_W[1] * 1.3), FIG_W[2]))
    path = joinpath(output_dir, "gamma_sensitivity.pdf")
    savefig(fig, path)
    @info "Saved $path"
end

# ── Tables ───────────────────────────────────────────────────────

function generate_tables(output_dir::String)
    @info "Generating table summaries..."
    generate_table1(output_dir)
    generate_table2(output_dir)
    generate_table3(output_dir)
    generate_table4(output_dir)
    generate_table5(output_dir)
    generate_table_exp6(output_dir)
    generate_table_exp7(output_dir)
    generate_table_exp8(output_dir)
end

function generate_table1(output_dir::String)
    df = load_csv(joinpath(expdir(1), "exp1_summary.csv"))
    if df === nothing; @warn "No Exp1 summary — skipping Tab.1"; return; end
    df_magi = load_csv(joinpath(MAGI_RESULTS, "lv_summary.csv"))

    rows = []
    for N in [10, 25, 40]
        for method in METHOD_ORDER
            if method == "MAGI" && df_magi !== nothing
                sub = filter(r -> r.N == N, df_magi)
            else
                sub = filter(r -> r.N == N && r.method == method, df)
            end
            if nrow(sub) == 0; continue; end
            ess_m, ess_se = mean_se(sub.ess_mean)
            rhat_vals = hasproperty(sub, :rhat_max) ? sub.rhat_max : fill(NaN, nrow(sub))
            rhat_m = mean(filter(!isnan, rhat_vals))
            push!(rows, (N=N, method=method,
                ess=fmt_mean_se(ess_m, ess_se), rhat=@sprintf("%.3f", rhat_m)))
        end
    end
    df_out = DataFrame(rows)
    path = joinpath(output_dir, "tab1_convergence.csv")
    CSV.write(path, df_out)
    @info "Saved $path"
end

function generate_table2(output_dir::String)
    df = load_csv(joinpath(expdir(2), "exp2_summary.csv"))
    if df === nothing; @warn "No Exp2 summary — skipping Tab.2"; return; end

    rows = []
    for ode in ["LV", "FN", "PST"]
        for method in METHOD_ORDER
            sub = filter(r -> r.ode == ode && r.method == method, df)
            if nrow(sub) == 0; continue; end
            rmsd_m, rmsd_se = mean_se(sub.rmsd)
            push!(rows, (ode=ode, method=method, rmsd=fmt_mean_se(rmsd_m, rmsd_se)))
        end
    end
    df_out = DataFrame(rows)
    path = joinpath(output_dir, "tab2_accuracy.csv")
    CSV.write(path, df_out)
    @info "Saved $path"
end

function generate_table3(output_dir::String)
    df = load_csv(joinpath(expdir(3), "exp3_summary.csv"))
    if df === nothing; @warn "No Exp3 summary — skipping Tab.3"; return; end

    rows = []
    kernel_order = ["RBF", "Matern52", "Laplace", "GenCauchy", "ExpPower"]
    for ode in unique(df.ode)
        for kernel in kernel_order
            for method in ["RFFGM", "GPGM"]
                sub = filter(r -> r.ode == ode && r.kernel == kernel && r.method == method, df)
                if nrow(sub) == 0; continue; end
                rmsd_m, rmsd_se = mean_se(sub.rmsd)
                ess_m, _ = mean_se(sub.ess_mean)
                push!(rows, (ode=ode, kernel=kernel, method=method,
                    rmsd=fmt_mean_se(rmsd_m, rmsd_se),
                    ess_mean=@sprintf("%.1f", ess_m)))
            end
        end
    end
    df_out = DataFrame(rows)
    path = joinpath(output_dir, "tab3_kernel_flexibility.csv")
    CSV.write(path, df_out)
    @info "Saved $path"
end

function generate_table4(output_dir::String)
    df = load_csv(joinpath(expdir(4), "exp4_summary.csv"))
    df_magi = load_csv(joinpath(MAGI_RESULTS, "lvc_summary.csv"))
    if df === nothing && df_magi === nothing
        @warn "No Exp4 data — skipping Tab.4"; return
    end

    rows = []
    for K in [2, 5, 10, 20]
        for method in METHOD_ORDER
            if method == "MAGI" && df_magi !== nothing
                sub = filter(r -> r.K == K, df_magi)
            elseif df !== nothing
                sub = filter(r -> r.K == K && r.method == method, df)
            else
                continue
            end
            if nrow(sub) == 0; continue; end
            rmsd_m, rmsd_se = mean_se(sub.rmsd)
            ess_m, _ = mean_se(sub.ess_mean)
            rhat_vals = hasproperty(sub, :rhat_max) ? sub.rhat_max : fill(NaN, nrow(sub))
            rhat_m = mean(filter(!isnan, rhat_vals))
            time_m, _ = mean_se(sub.time_sec)
            push!(rows, (K=K, method=method,
                rmsd=fmt_mean_se(rmsd_m, rmsd_se),
                ess_mean=@sprintf("%.1f", ess_m),
                rhat_max=@sprintf("%.3f", rhat_m),
                time_sec=@sprintf("%.1f", time_m)))
        end
    end
    df_out = DataFrame(rows)
    path = joinpath(output_dir, "tab4_scaling.csv")
    CSV.write(path, df_out)
    @info "Saved $path"
end

function generate_table5(output_dir::String)
    df = load_csv(joinpath(expdir(5), "exp5_summary.csv"))
    if df === nothing; @warn "No Exp5 summary — skipping Tab.5"; return; end

    rows = []
    for N in [50, 100, 200]
        for method in METHOD_ORDER
            sub = filter(r -> r.N == N && r.method == method, df)
            if nrow(sub) == 0; continue; end
            time_m, time_se = mean_se(sub.time_sec)
            ess_ps_m, ess_ps_se = mean_se(sub.ess_per_sec)
            push!(rows, (N=N, method=method,
                time_sec=fmt_mean_se(time_m, time_se),
                ess_per_sec=fmt_mean_se(ess_ps_m, ess_ps_se)))
        end
    end
    df_out = DataFrame(rows)
    path = joinpath(output_dir, "tab5_timing.csv")
    CSV.write(path, df_out)
    @info "Saved $path"
end

function generate_table_exp6(output_dir::String)
    df = load_csv(joinpath(expdir(6), "exp6_summary.csv"))
    if df === nothing; @warn "No Exp6 summary — skipping App.Tab.1"; return; end

    rows = []
    for method in METHOD_ORDER
        sub = filter(r -> r.method == method, df)
        if nrow(sub) == 0; continue; end
        for r in eachrow(sub)
            push!(rows, (method=r.method, seed=r.seed,
                rmsd=@sprintf("%.4f", r.rmsd),
                ess_mean=@sprintf("%.1f", r.ess_mean),
                time_sec=@sprintf("%.1f", r.time_sec)))
        end
    end
    df_out = DataFrame(rows)
    path = joinpath(output_dir, "apptab1_trajectory.csv")
    CSV.write(path, df_out)
    @info "Saved $path"
end

function generate_table_exp7(output_dir::String)
    df = load_csv(joinpath(expdir(7), "exp7_per_param.csv"))
    if df === nothing; @warn "No Exp7 per-param data — skipping App.Tab.2"; return; end

    rows = []
    params = unique(df.param_name)
    for method in ["RFFGM", "GPGM"]
        for param in params
            sub = filter(r -> r.method == method && r.param_name == param, df)
            if nrow(sub) == 0; continue; end
            rmsd_m, rmsd_se = mean_se(sub.param_rmsd)
            push!(rows, (method=method, param=param,
                rmsd=fmt_mean_se(rmsd_m, rmsd_se),
                mean_est=@sprintf("%.4f", mean(sub.param_mean)),
                true_val=@sprintf("%.4f", first(sub.param_true))))
        end
    end
    df_out = DataFrame(rows)
    path = joinpath(output_dir, "apptab2_pst_perparam.csv")
    CSV.write(path, df_out)
    @info "Saved $path"
end

function generate_table_exp8(output_dir::String)
    df = load_csv(joinpath(expdir(8), "exp8_summary.csv"))
    if df === nothing; @warn "No Exp8 summary — skipping App.Tab.3"; return; end

    rows = []
    kernels = unique(df.kernel)
    for kernel in kernels
        for method in METHOD_ORDER
            sub = filter(r -> r.kernel == kernel && r.method == method, df)
            if nrow(sub) == 0; continue; end
            ess_m, ess_se = mean_se(sub.ess_mean)
            rhat_vals = sub.rhat_max
            rhat_m = mean(filter(!isnan, rhat_vals))
            n_theta = count(c -> startswith(string(c), "theta_") &&
                endswith(string(c), "_mean"), names(sub))
            theta_strs = String[]
            for i in 1:n_theta
                col = Symbol("theta_$(i)_mean")
                if hasproperty(sub, col)
                    push!(theta_strs, @sprintf("%.3f", mean(sub[!, col])))
                end
            end
            push!(rows, (kernel=kernel, method=method,
                ess=fmt_mean_se(ess_m, ess_se),
                rhat=@sprintf("%.3f", rhat_m),
                theta_means=join(theta_strs, ", ")))
        end
    end
    df_out = DataFrame(rows)
    path = joinpath(output_dir, "apptab3_lynxhare.csv")
    CSV.write(path, df_out)
    @info "Saved $path"
end

# ── Main ─────────────────────────────────────────────────────────

function parse_args_main()
    s = ArgParseSettings(description="Generate PGM 2026 paper figures and tables")
    @add_arg_table! s begin
        "--output_dir"
            help = "Output directory for figures and tables"
            default = joinpath(@__DIR__, "figures")
    end
    return ArgParse.parse_args(s)
end

function main()
    args = parse_args_main()
    output_dir = args["output_dir"]
    mkpath(output_dir)

    @info "Output directory: $output_dir"

    # Generate figures (skip gracefully if data missing)
    plot_fig1_state_scatter(output_dir)
    plot_fig1_trajectory(output_dir)   # legacy (x, θ) view
    plot_fig1(output_dir)               # legacy convergence
    plot_fig2(output_dir)
    plot_appfig1(output_dir)
    plot_appfig2(output_dir)
    plot_appfig3(output_dir)
    plot_gamma_sensitivity(output_dir)

    # Generate tables
    generate_tables(output_dir)

    @info "Done! All outputs saved to $output_dir"
end

main()
