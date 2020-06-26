using PyCall
using Statistics
using TensorDecompositions

"Run Parafac Factorization with Mask"
function CP_decomposition(rank::Int64=5)
    # Init
    decomps = pyimport("tensorly.decomposition")
    cube = createCube()
    
    # Create Mask/Zero Out Data
    mask = .!(cube .=== nothing)
    cube[cube .=== nothing] .= 0
    
    # Convert Data Types
    cube = convert(Array{Float64,3}, cube)
    mask = convert(Array{Bool,3}, mask)
    
    # Run Factorizaton
    weights, factors = decomps.parafac(cube, rank, mask=mask, orthogonalise=true, init="random")
    return factors
end

"Re-compose tensor from CP decomposition"
function cp_reconstruct(factors::Array)
    lambdas = ones(size(factors[1], 2))
    tup = (factors[1], factors[2], factors[3])
    return compose(tup, lambdas)
end

"Calculate reconstruction error of two tensors with missing values"
function r2x(recon, orig)
    recon = replace(recon, nothing=>missing)
    orig = replace(orig, nothing=>missing)
    resid = recon .- orig
    itr_resid = skipmissing(resid)
    itr_orig = skipmissing(orig)
    return (1.0 - var(itr_resid)/var(itr_orig))
end