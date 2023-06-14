module Spass

include.(("SetUtils.jl", "MaxEnt.jl", "Lattice.jl", "Binomial.jl", "Discoverer.jl", "Precompile.jl"))

const spass = fit
const spass_fdr(args...; kw...) = spass(FDR, args...; kw...)
const spass_fwer(args...; kw...) = spass(FWER, args...; kw...)

export FWER, FDR, spass, spass_fwer, spass_fdr, patterns

end # module Spass
