"""
    QTCIOptions(; tolerance=1.0e-8, max_rank=64, max_sweeps=10)

Metadata-only placeholder for future QTCI configuration.
"""
Base.@kwdef struct QTCIOptions
    tolerance::Float64 = 1.0e-8
    max_rank::Int = 64
    max_sweeps::Int = 10
end

"""
    QTCIDiagnostics(; converged=false, sweeps=0, final_error=Inf)

Metadata-only placeholder for future QTCI diagnostics.
"""
Base.@kwdef struct QTCIDiagnostics
    converged::Bool = false
    sweeps::Int = 0
    final_error::Float64 = Inf
end

"""
    QTCIResultPlaceholder(options, diagnostics)

Placeholder container for future QTCI results.
"""
struct QTCIResultPlaceholder
    options::QTCIOptions
    diagnostics::QTCIDiagnostics
end
