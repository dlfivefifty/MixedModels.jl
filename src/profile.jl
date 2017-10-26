"""
    dropXcolumn!(m::LinearMixedModel, i::Integer)

Return `m` with column `i` of the fixed-effects model matrix, `X`, dropped

This operation changes dimensions of some blocks in `m.A` and `m.L`
FIXME: It looks as if A and L need to be generated anew.
"""
function dropXcolumn!(m::LinearMixedModel, i::Integer)
    trms = m.trms
    xpos = length(trms) - 1
    xtrm = trms[xpos] = reweight!(dropcolumn(trms[xpos], i), m.sqrtwts)
    for j in 1:xpos
        Axj = m.A[Block(xpos, j)] = xtrm'trms[j]
        m.L[Block(xpos, j)] = deepcopy(Axj)
    end
    Axp1x = m.A[Block(xpos + 1, xpos)] = trms[xpos + 1]'xtrm
    m.L[Block(xpos + 1, xpos)] = deepcopy(Axp1x)
    m
end
