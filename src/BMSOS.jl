module BMSOS

using LinearAlgebra

import NLopt
import TrigPolys
import AutoGrad

function fgradfft(U::AbstractArray, p::TrigPolys.TrigPoly; mapfn = Base.map)
    r, xs, xn, Up = get_Up(U, p)
    uB = mapfn(TrigPolys.evaluate, eachrow(Up))
    d = sum(u .^ 2 for u in uB) - TrigPolys.evaluate(p)
    g = reduce(hcat, mapfn(u -> TrigPolys.evaluateT(u .* d), uB))'
    return [g[:, 1:xs+1] g[:, 2*xs+2:3*xs+1]]
end

function ffft(U::AbstractArray, p::TrigPolys.TrigPoly; mapfn = Base.map)
    r, xs, xn, Up = get_Up(U, p)
    uB2 = mapfn(u -> TrigPolys.evaluate(u) .^ 2, eachrow(Up))
    d = sum(uB2) - TrigPolys.evaluate(p)
    return sum(d .^ 2)
end

function get_Up(U::AbstractArray, p::TrigPolys.TrigPoly)
    (r, _) = size(U)
    @assert p.n % 2 == 0 "Only multiple-of-4-degree polynomials supported!"
    xs = p.n ÷ 2
    xn = 2 * xs + 1
    Up = [U[:, 1:xs+1] zeros(r, xs) U[:, xs+2:xn] zeros(r, xs)]
    return r, xs, xn, Up
end

# Heuristic to find norm of initial point for LBFGS to converge better
function find_xinit_norm(r, n)
    pt = random_trig_poly(n)
    U = randn(r, n + 1)
    a = 1000 / norm(fgradfft(U, pt)[1, :])
    b = norm(fgradfft(U * a, pt))
    return 2 * a / b
end

function sos_decomp(
    p::TrigPolys.TrigPoly,
    r;
    method = :LD_LBFGS,
    init = 1e-8,
    max_eval = 1000,
    xtol_rel = 1e-7,
    mapfn = Base.map,
)
    # Computes the rank-r SOS decomposition of a polynomial
    @assert iseven(p.n) "p must have even degree!"
    xs = p.n ÷ 2
    xn = p.n + 1

    #### Slow but automatic-diffed version
    # function f(x)
    #     slices = [x[xn*(i-1)+1:xn*i] for i in 1:r]
    #     uB = sum([evaluate(pad_to(u, p.n)).^2 for u in slices])
    #     sum((uB - evaluate(p)).^2)
    # end
    # fgrad = AutoGrad.grad(f)
    #
    # fvals = Float64[]
    # function obj(x, grad)
    #     grad[:] = fgrad(x)
    #     last(append!(fvals, f(x)))
    # end

    #### Using manually written f(U) and ∇f(U) (3r+2 calls to FFT)
    # function obj(x::Vector, grad::Vector)
    #     U = reshape(x, r, xn)
    #     grad[:] = reshape(fgradfft(U, p; mapfn=mapfn),:)
    #     last(append!(fvals, ffft(U, p; mapfn=mapfn)))
    # end

    #### Even faster version (2r + 1 calls to FFT)
    fvals = Float64[]

    function obj_fast(x::Vector, grad::Vector)
        U = reshape(x, r, xn)
        r, xs, xn, Up = get_Up(U, p)
        uB = mapfn(evaluate, eachrow(Up))
        d = sum(u .^ 2 for u in uB) - evaluate(p)
        g = reduce(hcat, mapfn(u -> evaluateT(u .* d), uB))'
        grad[:] = reshape([g[:, 1:xs+1] g[:, 2*xs+2:3*xs+1]], :)
        return last(push!(fvals, sum(d .^ 2)))
    end

    opt = Opt(method, r * xn) #:LD_LBFGS, :LD_AUGLAG
    opt.maxeval = max_eval
    opt.xtol_rel = xtol_rel
    opt.min_objective = obj_fast
    xinit = randn(r * xn) * init
    stats = @timed (minf, minx, ret) = optimize(opt, xinit)
    Uopt = reshape(minx, r, xn)
    return (; fvals, Uopt, ret, stats)
end

function sos_opt(
    p::TrigPolys.TrigPoly,
    r;
    method = :LD_LBFGS,
    init = 1e-8,
    max_eval = 1000,
    xtol_rel = 1e-7,
)
    # Computes the minimum of p using a rank-r SOS decomposition
    @assert p.n % 2 == 0 "p must have even degree!"
    xs = p.n ÷ 2
    xn = p.n + 1
    nvars = r * xn + 1

    function f(x)
        slices = [x[xn*(i-1)+1:xn*i] for i in 1:r]
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
    opt.min_objective = obj

    xinit = randn(nvars) * init

    @time (minf, minx, ret) = NLopt.optimize(opt, xinit)
    @show ret

    Uopt = reshape(minx[1:nvars-1], r, xn)
    xopt = last(minx)
    return f, fgrad, fvals, xvals, Uopt, xopt, ret, minx
end

end
