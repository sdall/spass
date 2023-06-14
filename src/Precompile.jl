using PrecompileTools
using SparseArrays: SparseArrays

@setup_workload begin
    coeffs = (; alpha=0.5, min_support=1, max_factor_size=8, max_factor_width=100, max_expansions=20, max_discoveries=100)
    D = [trues(10, 10) falses(10, 10); falses(10, 10) trues(10, 10)] .| (rand(20, 20) .< 0.2)
    S = SparseArrays.sparse(D)
    B = map(x -> findall(!=(0), x) |> BitSet, eachrow(D))
    F = BitMatrix(D)
    y = rand(1:2, length(D))
    @compile_workload begin
        for Alg in (FDR, FWER), X in (D, F, S, B), Y in (y, nothing)
            fit(Alg, X, Y; coeffs...)
        end
    end
end
