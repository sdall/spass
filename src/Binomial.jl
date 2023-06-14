@fastmath function binomial_cdf_exact_lower(s, p, n)
    lg_true = log(p)
    lg_false = log(1 - p)
    b = zero(p)
    cdf = zero(p)
    for k in 1:s
        if k > 0
            b += log(n - k + 1) - log(k)
        end
        lg_pmf = k * lg_true + (n - k) * lg_false
        cdf += exp(lg_pmf + b)
    end
    cdf
end

@fastmath function binomial_cdf_exact_upper(s, p, n)
    lg_true = log(p)
    lg_false = log(1 - p)
    b = zero(p)
    cdf = zero(p)

    for k in 1:s
        b += log(n - k + 1) - log(k)
    end
    for k in (s + 1):n
        if k > 0
            b += log(n - k + 1) - log(k)
        end
        lg_pmf = k * lg_true + (n - k) * lg_false
        cdf += exp(lg_pmf + b)
    end
    cdfs
end

@fastmath binomial_cdf_exact(s, p, n) =
    if s < p * n
        binomial_cdf_exact_lower(s, p, n)
    elseif s > p * n
        binomial_cdf_exact_lower(n - s, 1 - p, n)
    else
        1.0
    end

@fastmath kl1(q, p) = q == 0 ? zero(q) : q * log(q / p)
@fastmath kl_bernoulli(q, p) = kl1(q, p) + kl1(1 - q, 1 - p)
@fastmath binomial_log_cdf_chernoff_impl(s, p, n) = -n * kl_bernoulli(s / n, p)
binomial_log_cdf_chernoff(s, p, n) =
    if s < n * p
        binomial_log_cdf_chernoff_impl(n - s, 1 - p, n)
    elseif s > n * p
        binomial_log_cdf_chernoff_impl(s, p, n)
    else
        zero(p)
    end

@fastmath binomial_cdf_chernoff(s, p, n) = exp(binomial_log_cdf_chernoff(s, p, n))
@fastmath binomial_log_cdf_exact(s, p, n) = log(binomial_cdf_exact(s, p, n))
@fastmath function binomial_log_cdf(s, p, n)
    n < 50 ? binomial_log_cdf_exact(s, clamp(p, 0.0, 1.0), n) : binomial_log_cdf_chernoff(s, clamp(p, 0.0, 1.0), n)
end
@fastmath binomial_cdf(s, p, n) = n < 50 ? binomial_cdf_exact(s, p, n) : binomial_cdf_chernoff(s, p, n)
