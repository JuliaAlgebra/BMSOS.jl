module TestRandom

import AutoGrad
import DynamicPolynomials
import MathOptInterface as MOI
import SumOfSquares as SOS
import TrigPolys
using BMSOS
using Test

import Random
Random.seed!(0)

function test_decomp(n = 20)
    p = TrigPolys.random_trig_poly(n)
    s = p.n * 2 + 1
    r = 2
    xs = p.n ÷ 2
    xn = 2 * xs + 1
    samples = [2π * (i - 1) / s for i in 1:s]
    B = reduce(hcat, [TrigPolys.basis(p.n, xi) for xi in samples])
    ps = reshape(p.(samples), 1, s)
    function f(U)
        Up = [U[:, 1:(xs+1)] zeros(r, xs) U[:, (xs+2):xn] zeros(r, xs)]
        return sum((sum((Up * B) .^ 2, dims = 1) - ps) .^ 2)
    end
    #f(U) = sum((sum((U*B).^2, dims=1)-ps).^2)
    fgrad = AutoGrad.grad(f)

    U = randn(r, xn)
    @test f(U) ≈ BMSOS.ffft(U, p)
    @test fgrad(U) ≈ BMSOS.fgradfft(U, p) * 4

    result = BMSOS.sos_decomp(p, rank = 2, init = BMSOS.find_xinit_norm(r, n))
    @test result.ret == :XTOL_REACHED
end

function test_sos_constraint()
    DynamicPolynomials.@polyvar x
    basis = SOS.SubBasis{SOS.Monomial}([x^0, x, x^2])
    set = SOS.moi_set(SOS.SOSCone(), basis, basis, basis)
    func = MOI.VectorAffineFunction(
        MOI.VectorAffineTerm{Float64}[],
        [1.0, 4.0, 4.0],
    )
    optimizer = BMSOS.Optimizer()
    @test MOI.supports_constraint(optimizer, typeof(func), typeof(set))
    MOI.add_constraint(optimizer, func, set)
    @test optimizer.poly == (1 + 2x)^2
    return
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

end

TestRandom.runtests()
