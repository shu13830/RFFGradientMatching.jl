# ANSI → HTML conversion functions (green code and reset code supported, horizontal scrollable container added)
function ansi_to_html(s::String)
    s = replace(s, "\033[32m" => "<span style=\"color: green;\">")
    s = replace(s, "\e[32m"   => "<span style=\"color: green;\">")
    s = replace(s, "\033[0m"  => "</span>")
    s = replace(s, "\e[0m"    => "</span>")
    return """
    <div style="overflow-x: auto; width: 100%;">
      <pre style="font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.2; white-space: pre;">
$s
      </pre>
    </div>
    """
end

function custom_progress_bar(i::Int, n_iter::Int, bar_length::Int=20)
    completed = Int(round((i / n_iter) * bar_length))
    bar = "[" * repeat("█", completed) * repeat(" ", bar_length - completed) * "]"
    progress_line = "Progress: $i / $n_iter $bar $(round(100 * i / n_iter, digits=1))%"
end

function logdens_history_plot_str(logdens_history::Vector{Float64})
    plot_str = UnicodePlots.lineplot(logdens_history; width=60, height=10, color=:green)
end

function compiled_output(i::Int, n_iter::Int, logdens_history::Vector{Float64})
    progress_line = custom_progress_bar(i, n_iter)
    plot_str = logdens_history_plot_str(logdens_history)
    ld = logdens_history[end]
    output = """
    $progress_line
    Latest logdensity: $(round(ld, digits=3))

    $plot_str
    """

    if is_terminal()
        # In CMD (terminal), screen is cleared with ANSI escape sequence and output is in plain text
        print("\033[H\033[J")
        println(output)
    elseif isdefined(Main, :IJulia) && isdefined(Main.IJulia, :clear_output)
        # In Jupyter Notebook, existing output is cleared and output is given in HTML after ANSI to HTML conversion
        html_output = ansi_to_html(output)
        Main.IJulia.clear_output()
        display("text/html", html_output)
    end
end

function is_terminal()::Bool
    return stdout isa Base.TTY
end
