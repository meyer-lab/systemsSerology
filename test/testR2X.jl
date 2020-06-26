
@testset "Test monotonically increasing R2X with more CP components" begin
    orig = systemsSerology.createCube()
    arr = zeros(10)

    for i = 1:8
        output = systemsSerology.CP_decomposition(i)
        reconstruct = systemsSerology.cp_reconstruct(output)
        arr[i] = systemsSerology.r2x(reconstruct, orig)

        @test arr[i] <= 1.0

        if i > 1
            @test arr[i] > arr[i-1]
        end
    end
end
