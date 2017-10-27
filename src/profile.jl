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
    LinearMixedModel(m.formula, trms, sqrtwts, A, LowerTriangular(L), osum)
end
