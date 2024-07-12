import MultivariatePolynomials as MP
import SemialgebraicSets
import MathOptInterface as MOI
import SumOfSquares

const DEFAULT = Dict{String,Any}(
    "rank" => 2,
    "init" => 1e-8,
    "mapfn" => Base.map,
    "algorithm" => :LD_LBFGS,
    # NLopt's parameters
    "xtol_rel" => 1e-7,
    "maxeval" => 0,
)

mutable struct Optimizer <: MOI.AbstractOptimizer
    poly::Union{Nothing,SA.AlgebraElement}
    options::Dict{String,Any}
    fvals::Union{Nothing,Vector{Float64}}
    Uopt::Union{Nothing,Matrix{Float64}}
    ret::Union{Nothing,Symbol}
    solve_time::Float64
    silent::Bool
    function Optimizer()
        return new(nothing, copy(DEFAULT), nothing, nothing, nothing, NaN, false)
    end
end

function MOI.supports(optimizer::Optimizer, attr::MOI.RawOptimizerAttribute)
    return haskey(optimizer.options, attr.name)
end
function MOI.get(optimizer::Optimizer, attr::MOI.RawOptimizerAttribute)
    if !MOI.supports(optimizer, attr)
        MOI.throw(optimizer, MOI.UnsupportedAttribute(attr))
    end
    return optimizer.options[attr.name]
end
function MOI.set(optimizer::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    if !MOI.supports(optimizer, attr)
        MOI.throw(optimizer, MOI.UnsupportedAttribute(attr))
    end
    optimizer.options[attr.name] = value
    return
end

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{<:SumOfSquares.SOSPolynomialSet{SemialgebraicSets.FullSpace}},
)
    return true
end
function MOI.add_constraint(
    optimizer::Optimizer,
    func::MOI.VectorAffineFunction{Float64},
    set::SumOfSquares.SOSPolynomialSet{SemialgebraicSets.FullSpace},
)
    if !isempty(func.terms)
        error("Nonconstant polynomials are not supported yet!")
    end
    # FIXME don't ignore `set.certificate`
    optimizer.poly = MB.algebra_element(func.constants, set.basis)
    return MOI.ConstraintIndex{typeof(func),typeof(set)}(0)
end

function MOI.empty!(optimizer::Optimizer)
    return optimizer.poly = nothing
end

MOI.is_empty(optimizer::Optimizer) = optimizer.poly === nothing

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

function MOI.optimize!(optimizer::Optimizer)
    options =
        Dict{Symbol,Any}(Symbol(key) => val for (key, val) in optimizer.options)
    start_time = time()
    ret = sos_decomp(optimizer.poly; options...)
    optimizer.solve_time = time() - start_time
    optimizer.fvals = ret.fvals
    optimizer.Uopt = ret.Uopt
    optimizer.ret = ret.ret
    return
end

MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec) = optimizer.solve_time

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    # TODO if we use the MOI API of NLopt, we can just redirect
    #      the call to it
    if optimizer.ret === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    else
        return MOI.OPTIMAL
    end
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return string(optimizer.ret)
end

function MOI.get(
    optimizer::Optimizer,
    ::SumOfSquares.GramMatrixAttribute,
    ::MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}},
)
    U = optimizer.Uopt
    Q = U' * U
    return SumOfSquares.GramMatrix(Q, optimizer.sampled.basis)
end

struct ErrorEstimates <: MOI.AbstractModelAttribute end
MOI.is_set_by_optimize(::ErrorEstimates) = true
function MOI.get(optimizer::Optimizer, ::ErrorEstimates)
    return optimizer.fvals
end
