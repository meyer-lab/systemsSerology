using DataFrames
using CSV

const dataDir = joinpath(dirname(pathof(systemsSerology)), "..", "data")

""" Import systems serology dataset. """
function importAlterMSB()
    dfF = CSV.read(joinpath(dataDir, "alter-MSB", "data-function.csv"))
    dfGP = CSV.read(joinpath(dataDir, "alter-MSB", "data-glycan-gp120.csv"))
    dfIGG = CSV.read(joinpath(dataDir, "alter-MSB", "data-luminex-igg.csv"))
    dfL = CSV.read(joinpath(dataDir, "alter-MSB", "data-luminex.csv"))
    dfMA = CSV.read(joinpath(dataDir, "alter-MSB", "meta-antigens.csv"))
    dfMD = CSV.read(joinpath(dataDir, "alter-MSB", "meta-detections.csv"))
    dfMG = CSV.read(joinpath(dataDir, "alter-MSB", "meta-glycans.csv"))
    dfMS = CSV.read(joinpath(dataDir, "alter-MSB", "meta-subjects.csv"))

    df = meltdf(dfL, view = true)
    newdfL = DataFrame(Rec = String[], Vir = String[], Sig = String[], Value = Float64[], Subject = Int64[])

    # Split column name into constituent parts
    for i = 1:size(df, 1)
        Ar = split(string(df.variable[i]), "."; limit = 3)
        if length(Ar) == 3
            push!(newdfL, [Ar[1], Ar[2], Ar[3], df.value[i], df.Column1[i]])
        else
            push!(newdfL, [Ar[1], Ar[2], "N/A", df.value[i], df.Column1[i]])
        end
    end

    return newdfL
end

function importLuminex()  #nearly same as importAlterMSB but does not separate into multiple columns
    dfL = CSV.read(joinpath(dataDir, "alter-MSB", "data-luminex.csv"))
    df = stack(dfL, view = true)
    rename!(df, [:Subject, :ColNames, :Value])

    # Convert FC column to strings
    df.ColNames = string.(df.ColNames)
    return df
end
