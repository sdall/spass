function create_mining_context(::Type{SetType}, max_factor_width) where SetType
    [SetType() for _ in 1:Threads.nthreads()],
    [SetType() for _ in 1:Threads.nthreads()],
    [MaxEntContext{SetType}() for _ in 1:Threads.nthreads()],
    [MaxEntContext{SetType}() for _ in 1:(max_factor_width + 1)]
end

@fastmath @inline log_adjustment_factor(k, N) = k * (1 + log(N) - log(k))
@fastmath @inline log_fwer_adjustment(alpha, K, N) = log(alpha) - log_adjustment_factor(K, N) # = log(a / (e^k (N/k)^k))

abstract type FWER end

"""
    fit(adjustment, x[, y]; kwargs...) 
    
Discovers a statistically significantly non-redundant set of patterns by leveraging a binomial redundancy test for a sequentially updated maximum entropy null-model.

# Arguments

- `adjustment::Union{FWER, FDR}`: Multiple hypothesis testing correction stategy.
- `x::Union{BitMatrix, AbstractMatrix{Bool}, Vector{BitSet}}`: Input dataset.
- `y::Vector{Integer}`: Class labels that indicate subgroups in `x`.

# Options

- `alpha::Float64`: Set the initial statistical significance level.
- `min_support::Integer`: Require a minimal support of each pattern.
- `max_factor_size::Integer`: Constraint the maximum number of patterns that each factor of the maximum entropy distribution can model. As inference complexity grows exponentially, the `max_factor_size` is bounded by [`MAX_MAXENT_FACTOR_SIZE`](@ref)=12.
- `max_factor_width::Integer`: Constraint the maximum number of singletons that each factor can model.
- `max_expansions::Integer`: Limit the number of search-space node-expansions per iteration. 
- `max_discoveries::Integer`: Terminate the algorithm after `max_discoveries` discoveries.
- `max_seconds::Float64`: Terminate the algorithm after approximately `max_seconds` seconds.

# Returns

Returns a factorized maximum entropy distribution [`MaxEnt`](@ref) which contains patterns, singletons, and estimated coefficients.
If `y` is specified, this function estimate and returns a distribution per group in `x` under a shared adjusted significance level.

Note: Extract patterns (discoveries) via [`patterns`](@ref) or the per-group patterns via `patterns.`.

# Example

```julia-repl
julia> using Spass: fit, FWER, FDR, patterns
julia> p = fit(FWER, X; alpha = 0.01)
julia> patterns(p)

julia> ps = fit(FDR, X, y; alpha = 0.01)
julia> patterns.(ps)
```

"""
function fit(::Type{FWER}, X; alpha=0.05, min_support=2, max_factor_size=8, max_factor_width=50, args...)
    @assert max_factor_size <= MAX_MAXENT_FACTOR_SIZE

    A, B, C, D = create_mining_context(BitSet, max_factor_width)

    L    = Lattice{Candidate{BitSet}}(X, x -> x.support)
    n, m = size(X, 1), length(L.singletons)
    p    = MaxEnt{BitSet,Float64}([s.support / n for s in L.singletons])

    log_alpha = log(alpha)
    layer     = 2

    @inline isforbidden(x) = isforbidden_ts!(p, deepcopy(x.set), max_factor_size, max_factor_width, A)
    @inline function score(x)::Float64
        if x.support < min_support || isforbidden(x)
            0.0
        else
            E = expectation_ts!(p, x.set, A, B, C)
            pv = -binomial_log_cdf(x.support, E, n)
            th = log_alpha::Float64 - log_adjustment_factor(max(length(x.set), layer::Int), m)
            (pv >= -th) ? pv : 0.0
        end
    end

    function report(x)
        layer = max(layer::Int, length(x.set))
        if x.score >= -log_fwer_adjustment(alpha, layer::Int, m)
            insert_pattern!(p, x.support / n, x.set, max_factor_size, max_factor_width, D)
        else
            false
        end
    end

    discover_patterns!(L, score, isforbidden, report; args...)
    p
end

function fit(alg::Type{FWER}, X, y; alpha=0.05, min_support=2, max_factor_size=8, max_factor_width=50, args...)
    @assert max_factor_size <= MAX_MAXENT_FACTOR_SIZE

    if y === nothing
        return fit(alg, X; alpha=alpha,
                   min_support=min_support, max_factor_size=max_factor_size,
                   max_factor_width=max_factor_width, args...)
    end

    A, B, C, D = create_mining_context(BitSet, max_factor_width)

    masks = [BitSet(findall((y .== i))) for i in unique(y)]
    L     = Lattice{Candidate{BitSet}}(X, x -> x.support)
    n, m  = length.(masks), length(L.singletons)

    fr(s, j) = intersection_size(s.rows, masks[j]) / n[j]
    Pr       = MaxEnt{BitSet,Float64}
    p        = Pr[Pr(fr.(L.singletons, j)) for j in eachindex(n)]

    layer     = 2
    log_alpha = log(alpha)

    isforbidden(x) = isforbidden_ts!(p, x.set, max_factor_size, max_factor_width, B)
    function score(x)::Float64
        if x.support < min_support || isforbidden(x)
            0.0
        else
            tid = Threads.threadid()
            sum(eachindex(p)) do i
                E  = expectation_ts!(p[i], x.set, A, B, C; tid=tid)
                q  = intersection_size(x.rows, masks[i])
                pv = -binomial_log_cdf(q, E, n[i])
                th = log_alpha::Float64 - log_adjustment_factor(max(length(x.set), layer::Int), m)

                (pv >= -th) ? pv : 0.0
            end
        end
    end
    function report(x)
        layer = max(layer::Int, length(x.set))
        tid   = Threads.threadid()
        mapreduce(|, eachindex(p)) do i
            E = expectation_ts!(p[i], x.set, A, B, C; tid=tid)
            q = intersection_size(x.rows, masks[i])

            pv = -binomial_log_cdf(q, E, n[i])
            th = log_alpha::Float64 - log_adjustment_factor(max(length(x.set), layer::Int), m)

            (pv >= -th) && insert_pattern!(p[i], q / n[i], x.set, max_factor_size, max_factor_width, D)
        end
    end
    discover_patterns!(L, score, isforbidden, report; args...)
    p
end

@inline @fastmath Xi(k::Int, alpha, b_0) = 6 / (pi^2 * k^2) * (alpha / b_0) / (1 + log(k))

mutable struct LORD
    alpha   :: Float64
    w0      :: Float64
    b0      :: Float64
    alpha_i :: Float64
    w       :: Float64
    w_tau   :: Float64
    i       :: Int
    tau     :: Int

    function LORD(alpha=0.05, factor=0.5)
        w0 = alpha * factor
        b0 = alpha - w0
        alpha_i = Xi(1, alpha, b0) * w0
        w_tau = w = w0 - alpha_i + b0
        new(alpha, w0, b0, alpha_i, w, w_tau, 1, 0)
    end
end

@fastmath function test(self::LORD, pvalue)
    reject = pvalue < self.alpha_i
    reject, pvalue, self.alpha_i

    if reject
        self.tau   = self.i
        self.w_tau = self.w
    end

    self.i       = self.i + 1
    self.alpha_i = Xi(self.i, self.alpha, self.b0) * self.w_tau
    self.w       = self.w - self.alpha_i + reject * self.b0

    reject
end

abstract type FDR end

function fit(alg::Type{FDR}, X; alpha=0.05, min_support=2, max_factor_size=8, max_factor_width=50, args...)
    @assert max_factor_size <= MAX_MAXENT_FACTOR_SIZE

    A, B, C, D = create_mining_context(BitSet, max_factor_width)

    n          = size(X, 1)
    L          = Lattice{Candidate{BitSet}}(X, x -> x.support)
    p          = MaxEnt{BitSet,Float64}([s.support / n for s in L.singletons])
    log_alpha  = -log(alpha)
    adjustment = LORD(alpha)

    isforbidden(x) = isforbidden_ts!(p, x.set, max_factor_size, max_factor_width, A)
    discover_patterns!(L, x -> if x.support < min_support || isforbidden(x)
                           0.0
                       else
                           E = expectation_ts!(p, x.set, A, B, C)
                           # pvalues are proportional to information gain, which is the score that is used rank candidates before testing the next.
                           pv = -binomial_log_cdf(x.support, E, n)
                           # discards hopeless hypotheses
                           pv > log_alpha ? pv : 0.0
                       end,
                       isforbidden,
                       x -> test(adjustment, exp(-x.score)) &&
                           insert_pattern!(p, x.support / n, x.set, max_factor_size, max_factor_width, D)
                       ; args...)
    p
end

function fit(alg::Type{FDR}, X, y; alpha=0.05, min_support=2, max_factor_size=8, max_factor_width=50, args...)
    @assert max_factor_size <= MAX_MAXENT_FACTOR_SIZE

    if isnothing(y) || length(y) == 0
        return fit(alg, X; alpha=alpha, min_support=min_support, max_factor_size=max_factor_size,
                   max_factor_width=max_factor_width, args...)
    end

    A, B, C, D = create_mining_context(BitSet, max_factor_width)

    masks = [BitSet(findall((y .== i))) for i in unique(y)]
    n     = length.(masks)
    L     = Lattice{Candidate{BitSet}}(X, x -> x.support)

    fr(s, j) = intersection_size(s.rows, masks[j]) / n[j]
    Pr       = MaxEnt{BitSet,Float64}
    p        = Pr[Pr(fr.(L.singletons, j)) for j in eachindex(n)]

    log_alpha  = -log(alpha)
    adjustment = LORD(alpha)

    isforbidden(x) = isforbidden_ts!(p, x.set, max_factor_size, max_factor_width, A)
    score(x) =
        if x.support < min_support || isforbidden(x)
            0.0
        else
            tid = Threads.threadid()
            sum(eachindex(p)) do i
                E = expectation_ts!(p[i], x.set, A, B, C; tid=tid)
                q = intersection_size(x.rows, masks[i])
                pv = -binomial_log_cdf(q, E, n[i])
                (pv >= -log_alpha) ? pv : 0.0
            end
        end
    function report(x)
        tid = Threads.threadid()
        mapreduce(|, eachindex(p); init=false) do i
            E = expectation_ts!(p[i], x.set, A, B, C; tid=tid)
            q = intersection_size(x.rows, masks[i])
            b = binomial_log_cdf(q, E, n[i])

            test(adjustment, exp(b)) && insert_pattern!(p[i], q / n[i], x.set, max_factor_size, max_factor_width, D)
        end
    end
    discover_patterns!(L, score, isforbidden, report; args...)
    p
end
