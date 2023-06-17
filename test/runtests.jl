module TestRandom

import AutoGrad
import TrigPolys
using BMSOS
using Test

function test_opt(n = 1000)
    p = TrigPolys.random_trig_poly(n)
    s = p.n * 2 + 1
    r = 2
    xs = p.n ÷ 2
    xn = 2 * xs + 1
    samples = [2π * (i - 1) / s for i in 1:s]
    B = reduce(hcat, [TrigPolys.basis(p.n, xi) for xi in samples])
    ps = reshape(p.(samples), 1, s)
    function f(U)
        Up = [U[:, 1:xs+1] zeros(r, xs) U[:, xs+2:xn] zeros(r, xs)]
        return sum((sum((Up * B) .^ 2, dims = 1) - ps) .^ 2)
    end
    #f(U) = sum((sum((U*B).^2, dims=1)-ps).^2)
    fgrad = AutoGrad.grad(f)

    U = randn(r, xn)
    @test f(U) ≈ BMSOS.ffft(U, p)
    @test fgrad(U) ≈ BMSOS.fgradfft(U, p) * 4
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
