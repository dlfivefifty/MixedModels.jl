"""
    dropXcolumn(m::LinearMixedModel, i::Integer)

Return a copy of `m` with column `i` of the fixed-effects model matrix, `X`, dropped
"""
function dropXcolumn(m::LinearMixedModel, i::Integer)
    trms = copy.(m.trms)
    xpos = length(trms) - 1
    trms[xpos] = dropcolumn(trms[xpos], i)
    sqrtwts = copy(m.sqrtwts)
    reweight!.(trms, [sqrtwts])
    A, L = createAL(trms)   
    osum = copy(m.optsum)
    copy!(osum.initial, m.optsum.final)
    osum.initial_step .= osum.initial_step ./ 4.0
    LinearMixedModel(m.formula, trms, sqrtwts, A, LowerTriangular(L), osum)
end

struct ProfileSlice{T<:AbstractFloat}
    zeta::Vector{T}
    beta::Vector{T}
    beta1::Matrix{T}
    theta::Matrix{T}
end

struct Profile{T<:AbstractFloat}
    beta::Vector{T}
    sigma::Vector{T}
    slices::Vector{ProfileSlice{T}}
end

function profileBeta(m::LinearMixedModel{T}) where T
    X = m.trms[end - 1].x
    beta = fixef(m)
    se = stderr(m)
    p = length(beta)
    d0 = deviance(m)
    y0 = copy(model_response(m))
    slices = sizehint!(ProfileSlice{T}[], p)
    zvals = -4.0:0.5:4.0
    lzv = length(zvals)
    for i in 1:p
        m1 = dropXcolumn(m, i)
        zeta = Vector{T}(lzv)
        ff = Matrix{T}((p - 1), lzv)
        th = Matrix{T}(sum(nθ, m.trms), lzv)
        for (j, z) in enumerate(zvals)
            if iszero(z)
                zeta[j] = zero(T)
                copy!(view(ff, :, j), deleteat!(copy(beta), i))
                getθ!(view(th, :, j), m)
            else
                refit!(m1, y0 .- (beta[i] + z * se[i]) .* view(X, :, i))
                zeta[j] = sign(z) * sqrt(deviance(m1) - d0)
                fixef!(view(ff, :, j), m1)
                getθ!(view(th, :, j), m1)
            end
        end
        push!(slices, ProfileSlice(zeta, beta[i] .+ se[i] .* zvals, ff', th'))
    end
    slices
end

