module BMSOS

using LinearAlgebra

import NLopt
import TrigPolys
import AutoGrad

"""
    fgradfft(U::AbstractArray, p::TrigPolys.TrigPoly; mapfn = Base.map)

Evaluates the gradient of `||b'U'Ub - p||^2` with respect to `U` where `b` is
the polynomial basis.
"""
function fgradfft(U::AbstractMatrix, p::TrigPolys.TrigPoly; mapfn = Base.map)
    _, xs, _, Up = get_Up(U, p)
    uB = mapfn(TrigPolys.evaluate, eachrow(Up))
    d = sum(u .^ 2 for u in uB) - TrigPolys.evaluate(p)
    g = reduce(hcat, mapfn(u -> TrigPolys.evaluateT(u .* d), uB))'
    return [g[:, 1:xs+1] g[:, 2*xs+2:3*xs+1]]
end

"""
    ffft(U::AbstractArray, p::TrigPolys.TrigPoly; mapfn = Base.map)

Evaluates `||b'U'Ub - p||^2` where `b` is the polynomial basis.
"""
function ffft(U::AbstractArray, p::TrigPolys.TrigPoly; mapfn = Base.map)
    _, _, _, Up = get_Up(U, p)
    uB2 = mapfn(u -> TrigPolys.evaluate(u) .^ 2, eachrow(Up))
    d = sum(uB2) - TrigPolys.evaluate(p)
    return sum(d .^ 2)
end

"""
    value_and_gradient!(grad, U::AbstractMatrix, p::TrigPolys.TrigPoly; mapfn = Base.map)

Evaluates `||b'U'Ub - p||^2` where `b` is the polynomial basis and set its
gradient to `grad` with `2r + 1` calls to FFT instead of `3r + 2` if
`ffft` and `fgradfft` are called separately.
"""
function value_and_gradient!(grad, U::AbstractMatrix, p::TrigPolys.TrigPoly; mapfn = Base.map)
    _, xs, _, Up = get_Up(U, p)
    uB = mapfn(TrigPolys.evaluate, eachrow(Up))
    d = sum(u .^ 2 for u in uB) - TrigPolys.evaluate(p)
    g = reduce(hcat, mapfn(u -> TrigPolys.evaluateT(u .* d), uB))'
    grad[:] = reshape([g[:, 1:xs+1] g[:, 2*xs+2:3*xs+1]], :)
    return sum(d .^ 2)
end

"""
    get_Up(U::AbstractArray, p::TrigPolys.TrigPoly)

Given a matrix of the form
```
[
    u11' u21'
    u12' u21'
    ...
    u1r' u21'
]
```
returns
```
[
    u11' 0
    u12' 0
    ...
    u1r' 0
    0    u21'
    0    u21'
    ...
    0    u21'
]
```
"""
function get_Up(U::AbstractArray, p::TrigPolys.TrigPoly)
    r, n = size(U)
    @assert p.n % 2 == 0 "Only multiple-of-4-degree polynomials supported!"
    xs = p.n ÷ 2
    # FIXME why not `2 * xs + 2` ?
    xn = 2 * xs + 1
    @assert xn == n "Inconsistent matrix dimension"
    Up = [U[:, 1:xs+1] zeros(r, xs) U[:, xs+2:xn] zeros(r, xs)]
    return r, xs, xn, Up
end

"""
    find_xinit_norm(r, n)

Heuristic to find norm of initial point for LBFGS to converge better
"""
function find_xinit_norm(r, n)
    pt = TrigPolys.random_trig_poly(n)
    U = randn(r, n + 1)
    a = 1000 / norm(fgradfft(U, pt)[1, :])
    b = norm(fgradfft(U * a, pt))
    return 2 * a / b
end

function _opt()
end

"""
    function sos_decomp(
        p::TrigPolys.TrigPoly,
        r;
        method = :LD_LBFGS,
        init = 1e-8,
        max_eval = 1000,
        xtol_rel = 1e-7,
        mapfn = Base.map,
    )

Computes the Sum-of-Squares decomposition of the projection of `p` to the
Sum-of-Squares cone using [LYP23].

[LYP22] Legat, Benoît and Yuan Chenyang and Parrilo, Pablo A.
*Low-Rank Univariate Sum of Squares Has No Spurious Local Minima*
2205.11466, arXiv, math.OC
"""
function sos_decomp(
    p::TrigPolys.TrigPoly;
    rank = 2,
    algorithm = :LD_LBFGS,
    init = 1e-8,
    maxeval = 1000,
    xtol_rel = 1e-7,
    mapfn = Base.map,
)
    # Computes the rank-r SOS decomposition of a polynomial
    @assert iseven(p.n) "p must have even degree!"
    xn = p.n + 1

    fvals = Float64[]

    function obj_fast(x::Vector, grad::Vector)
        U = reshape(x, rank, xn)
        val = value_and_gradient!(grad, U, p; mapfn)
        push!(fvals, val)
        return val
    end

    opt = NLopt.Opt(algorithm, rank * xn) #:LD_LBFGS, :LD_AUGLAG
    if !isnan(xtol_rel)
        NLopt.xtol_rel!(opt, xtol_rel)
    end
    NLopt.maxeval!(opt, maxeval)
    NLopt.min_objective!(opt, obj_fast)
    xinit = randn(rank * xn) * init
    stats = @timed (minf, minx, ret) = NLopt.optimize(opt, xinit)
    Uopt = reshape(minx, rank, xn)
    return (; fvals, Uopt, ret, stats)
end

function sos_opt(
    p::TrigPolys.TrigPoly;
    rank = 2,
    method = :LD_LBFGS,
    init = 1e-8,
    max_eval = 1000,
    xtol_rel = 1e-7,
)
    # Computes the minimum of p using a rank-r SOS decomposition
    @assert p.n % 2 == 0 "p must have even degree!"
    xn = p.n + 1
    nvars = rank * xn + 1

    function f(x)
        slices = [x[xn*(i-1)+1:xn*i] for i in 1:rank]
        gam = last(x)
        uB = sum([evaluate(pad_to(u, p.n)) .^ 2 for u in slices])
        return sum((uB - evaluate(p) .+ gam) .^ 2)
    end
    fgrad = AutoGrad.grad(f)

    fvals = Float64[]
    xvals = Float64[]

    function obj(x, grad)
        grad[:] = fgrad(x)
        append!(xvals, last(x))
        return last(append!(fvals, f(x)))
    end

    function obj2(x, grad)
        grad[:] = [zeros(nvars - 1); 1]
        return last(x)
    end

    opt = NLopt.Opt(method, nvars)
    opt.maxeval = max_eval
    opt.xtol_rel = xtol_rel
    #equality_constraint!(opt, eqconst, 1e-7)
    NLopt.min_objective!(opt, obj)

    xinit = randn(nvars) * init

    @time (minf, minx, ret) = NLopt.optimize(opt, xinit)
    @show ret

    Uopt = reshape(minx[1:nvars-1], rank, xn)
    xopt = last(minx)
    return f, fgrad, fvals, xvals, Uopt, xopt, ret, minx
end

include("MOI_wrapper.jl")

end
