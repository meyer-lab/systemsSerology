"Test monotonically increasing R2X with more CP components"
function test_CP_R2X()
    orig = createCube()
    arr = zeros(10)
    for i = 1:10
        output = CP_decomposition(i)
        reconstruct = cp_reconstruct(output)
        arr[i] = r2x(reconstruct, orig)
        if i > 1
            @assert arr[i] > arr[i-1]
        end
    end
end
