using TreeTools
using StaticArrays
using NearestNeighbors
reload("HeightOrderedGrid")


function means_hogrid(points_svec, grid)
    #n = Vector{Vector{SVector{3,Float64}}}(length(points_svec))
    means = fill(zeros(SVector{3,Float64}), length(points_svec))
    for i ∈ 1:length(points_svec)
        @inbounds p = points_svec[i]
        neighbors = HeightOrderedGrid.inrange(grid, p, 0.4)
        j = 0
        #n[i] = Vector{SVector{3,Float64}}()
        #for p2 ∈ neighbors
            #push!(n[i], p2-p)
        #    j += 1
        #    @inbounds means[i] += p2 - p
            #if vecnorm(p2 - p) > 0.4
            #    println("?")
            #end
        #end
        @inbounds means[i] /= j
    end
    return means
end

function means_kdtree(points_svec, tree)
    #n = Vector{Vector{SVector{3,Float64}}}(length(points_svec))
    means = fill(zeros(SVector{3,Float64}), length(points_svec))
    for i ∈ 1:length(points_svec)
        @inbounds p = points_svec[i]
        @inbounds neighbors = inrange(tree, [p], 0.4)[1]
        j = 0
        #n[i] = Vector{SVector{3,Float64}}()
        for i2 ∈ neighbors
            #push!(n[i], points_svec[i2] - p)
            j += 1
            @inbounds means[i] += points_svec[i2] - p
            #if vecnorm(points_svec[i2] - p) > 0.4
            #    println("!")
            #end
        end
        @inbounds means[i] /= j
    end
    return means
end


(agl, dtm, points, color, nor, rn, intensity) = TreeTools.load_points_hdf5("")

points_svec = reinterpret(SVector{3,Float64}, points, (size(points,2),))


print("Constructing HOGrid: ")
HeightOrderedGrid.HOGrid(points_svec, 0.4)
@time grid = HeightOrderedGrid.HOGrid(points_svec, 0.4)

print("Querying HOGrid:     ")
means_hogrid(points_svec, grid)
@time means_1 = means_hogrid(points_svec, grid)

println()
print("Constructing KDTree: ")
KDTree(points)
@time tree = KDTree(points)

print("Querying KDTree:     ")
means_kdtree(points_svec, tree)
@time means_2 = means_kdtree(points_svec, tree)

println()
