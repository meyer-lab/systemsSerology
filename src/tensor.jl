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
    weights, factors = decomps.parafac(cube, rank, mask=mask)
    return factors
end

"Run Tucker Factorization with Mask"
function tucker_decomposition(rank::Tuple=(2,2,2))
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
    core, factors = decomps.tucker(cube, rank, mask=mask)
    return (core, factors)
end

"Re-compose tensor from CP decomposition"
function cp_reconstruct(factors::Array)
    lambdas = ones(1, size(factors[1], 2))
    lambdas = vec(lambdas)
    tup = (factors[1], factors[2], factors[3])
    dest = ones(181, 22, 41)
    compose!(dest, tup, lambdas)
end

"Re-compose tensor from Tucker decomposition"
function tucker_reconstruct(output)
    decomp = Tucker(output[2], output[1])
    dest = ones(181, 22, 41)
    reconstruct = compose!(dest, decomp)
    return reconstruct
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