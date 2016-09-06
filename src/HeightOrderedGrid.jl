module HeightOrderedGrid

using StaticArrays

export HOGrid

"""
    immutable HOGrid{T, M <: AbstractMatrix}

A spatial index suitable for 3D data clustered about a plane (so-called
2.5D data) such as large-scale point clouds of terrain. Allows for fast querying
of points withing a radius of a specified location.
"""
immutable HOGrid{T}
    grid::Matrix{Vector{SVector{3,T}}}
    x0::T
    y0::T
    spacing::T
end

"""
    HOGrid(points::Vector{SVector{3,T}}, gridspacing) -> grid

Instantiate a height-ordered grid of `points` with grid spacing `gridspacing`.
During the process, the `points` will be copied and reordered to maximise cache
coherency on spatial querying.
"""
function HOGrid{T}(points::Vector{SVector{3,T}}, gridspacing)
    if length(points) == 0
        error("Cannot create HOGrid for zero points")
    end

    (xmin, xmax, ymin, ymax) = bounds(points)
    xmin -= eps(xmin) # make sure we never fall into grid 0
    ymin -= eps(ymin)

    n_x = Int(cld(xmax - xmin, gridspacing))
    n_y = Int(cld(ymax - ymin, gridspacing))

    grid = Matrix{Vector{SVector{3,T}}}((n_x, n_y))
    for i = 1:n_x
        for j = 1:n_y
            grid[i,j] = Vector{SVector{3,T}}() # Should this be done sparsely?
        end
    end

    # Stash each point into the correct grid cell
    for p ∈ points
        @inbounds x = p[1]
        @inbounds y = p[2]

        grid_x = Int(cld(x - xmin, gridspacing))
        grid_y = Int(cld(y - ymin, gridspacing))

        @inbounds push!(grid[grid_x, grid_y], p)
    end

    # Sort all the grid cells on height
    for g ∈ grid
        sort!(g; lt = height_less)
    end

    return HOGrid(grid, xmin, ymin, gridspacing)
end


immutable SpatialQuery{T}
    grid::HOGrid{T}
    query_point::SVector{3,T}
    radius::T
    gx_min::Int
    gx_max::Int
    gy_min::Int
    gy_max::Int
end

function SpatialQuery{T}(grid::HOGrid{T}, query_point::SVector{3,T}, radius::T)
    r² = radius * radius

    @inbounds qx = query_point[1]
    @inbounds qy = query_point[2]

    # Make a square cutoff
    xmin = qx - radius
    xmax = qx + radius
    ymin = qy - radius
    ymax = qy + radius

    # Make a square grid
    gx_min = Int(cld(xmin - grid.x0, grid.spacing))
    if gx_min < 1
        gx_min = 1
    end
    gx_max = Int(cld(xmax - grid.x0, grid.spacing))
    if gx_max > size(grid.grid, 1)
        gx_max = size(grid.grid, 1)
    end

    gy_min = Int(cld(ymin - grid.y0, grid.spacing))
    if gy_min < 1
        gy_min = 1
    end
    gy_max = Int(cld(ymax - grid.y0, grid.spacing))
    if gy_max > size(grid.grid, 2)
        gy_max = size(grid.grid, 2)
    end

    return SpatialQuery(grid, query_point, radius, gx_min, gx_max, gy_min, gy_max)
end

@inline height{T <: Number}(p::SVector{3,T}) = @inbounds return p[3]
@inline height{T <: Number}(h::T) = h

function Base.start{T}(sq::SpatialQuery{T})
    gx = sq.gx_min
    gy = sq.gy_min
    r² = sq.radius * sq.radius

    @inbounds qx = sq.query_point[1]
    @inbounds qy = sq.query_point[2]
    @inbounds qz = sq.query_point[3]

    while true
        if gy > sq.gy_max
            break
        end

        # Fit a sphere and find the vertical range to query, if any
        if qy > sq.grid.y0 + gy * sq.grid.spacing
            dy = qy - (sq.grid.y0 + gy * sq.grid.spacing)
        elseif qy >= sq.grid.y0 + (gy - 1) * sq.grid.spacing
            dy = zero(T)
        else
            dy = (sq.grid.y0 + (gy - 1) * sq.grid.spacing) - qy
        end

        while true
            if gx > sq.gx_max
                gx = sq.gx_min
                break
            end

            # Fit a sphere and find the vertical range to query, if any
            if qx > sq.grid.x0 + gx * sq.grid.spacing
                dx = qx - (sq.grid.x0 + gx * sq.grid.spacing)
            elseif qx >= sq.grid.x0 + (gx - 1) * sq.grid.spacing
                dx = zero(T)
            else
                dx = (sq.grid.x0 + (gx - 1) * sq.grid.spacing) - qx
            end

            h² = dx*dx + dy*dy
            if h² > r²
                gx += 1
                continue
            end
            dz = sqrt(r² - h²)

            # Now find the first element in this range
            @inbounds cell = sq.grid.grid[gx, gy]
            i = searchsortedfirst(cell, qz - dz, Base.By(height))

            # Now loop until we find one
            while true
                if i > size(cell, 1)
                    break
                end

                @inbounds p = cell[i]
                diff = p - sq.query_point
                if dot(diff, diff) <= r²
                    return (gx, gy, i, dz)
                end

                if height(p) - qz > dz
                    break
                end

                i += 1
            end

            gx += 1
        end

        gy += 1
    end

    return (gx, gy, i, dz)
end


function Base.next{T}(sq::SpatialQuery{T}, state::Tuple{Int,Int,Int,T})
    (gx, gy, i, dz) = state
    r² = sq.radius * sq.radius
    @inbounds qx = sq.query_point[1]
    @inbounds qy = sq.query_point[2]
    @inbounds qz = sq.query_point[3]

    # If we are calling next, then gx, gy, i is a valid state
    @inbounds cell = sq.grid.grid[gx, gy]
    @inbounds p_now = cell[i]

    # increment i and check the same cell before continuing
    while true
        i += 1

        if i > size(cell, 1)
            break
        end

        @inbounds p = cell[i]
        diff = p - sq.query_point
        if dot(diff, diff) <= r²
            return (p_now, (gx, gy, i, dz))
        end

        if height(p) - qz > dz
            break
        end
    end

    # increment gx and check other cells
    gx += 1

    while true
        if gy > sq.gy_max
            break
        end

        # Fit a sphere and find the vertical range to query, if any
        if qy > sq.grid.y0 + gy * sq.grid.spacing
            dy = qy - (sq.grid.y0 + gy * sq.grid.spacing)
        elseif qy >= sq.grid.y0 + (gy - 1) * sq.grid.spacing
            dy = zero(T)
        else
            dy = (sq.grid.y0 + (gy - 1) * sq.grid.spacing) - qy
        end

        while true
            if gx > sq.gx_max
                gx = sq.gx_min
                break
            end

            # Fit a sphere and find the vertical range to query, if any
            if qx > sq.grid.x0 + gx * sq.grid.spacing
                dx = qx - (sq.grid.x0 + gx * sq.grid.spacing)
            elseif qx >= sq.grid.x0 + (gx - 1) * sq.grid.spacing
                dx = zero(T)
            else
                dx = (sq.grid.x0 + (gx - 1) * sq.grid.spacing) - qx
            end

            h² = dx*dx + dy*dy
            if h² > r²
                gx += 1
                continue
            end
            dz = sqrt(r² - h²)

            # Now find the first element in this range
            @inbounds cell = sq.grid.grid[gx, gy]
            i = searchsortedfirst(cell, qz - dz, Base.By(height))

            # Now loop until we find one
            while true
                if i > size(cell, 1)
                    break
                end

                @inbounds p = cell[i]
                diff = p - sq.query_point
                if dot(diff, diff) <= r²
                    return (p_now, (gx, gy, i, dz))
                end

                if height(p) - qz > dz
                    break
                end

                i += 1
            end

            gx += 1
        end

        gy += 1
    end

    return (p_now, (gx, gy, i, dz))
end

Base.done{T}(sq::SpatialQuery{T}, state::Tuple{Int,Int,Int,T}) = state[2] > sq.gy_max












immutable SpatialIterator{T, ZMin, ZMax}
    grid_cells::Vector{Vector{SVector{3,T}}}
    dz::Vector{T}
    query_point::SVector{3,T}
    radius::T
    zmin::ZMin
    zmax::ZMax
end

# Returns an spatial iterator for getting points within a radius of a point
@inline function inrange{T}(grid::HOGrid{T}, query_point, radius; zmin = nothing, zmax = nothing)
    grid_cells = Vector{Vector{SVector{3,T}}}()
    dz = Vector{T}()
    r² = radius * radius

    @inbounds qx = query_point[1]
    @inbounds qy = query_point[2]

    # Make a square cutoff
    xmin = qx - radius
    xmax = qx + radius
    ymin = qy - radius
    ymax = qy + radius

    # Make a square grid
    g_xmin = Int(cld(xmin - grid.x0, grid.spacing))
    if g_xmin < 1
        g_xmin = 1
    end
    g_xmax = Int(cld(xmax - grid.x0, grid.spacing))
    if g_xmax > size(grid.grid, 1)
        g_xmax = size(grid.grid, 1)
    end

    g_ymin = Int(cld(ymin - grid.y0, grid.spacing))
    if g_ymin < 1
        g_ymin = 1
    end
    g_ymax = Int(cld(ymax - grid.y0, grid.spacing))
    if g_ymax > size(grid.grid, 2)
        g_ymax = size(grid.grid, 2)
    end

    # Refine the square grid into a cicular one and compose the cells
    for g_x ∈ g_xmin:g_xmax
        # Check if qx is in this cell, or to the left or right
        if qx > grid.x0 + g_x * grid.spacing
            dx = qx - (grid.x0 + g_x * grid.spacing)
        elseif qx >= grid.x0 + (g_x - 1) * grid.spacing
            dx = zero(T)
        else
            dx = (grid.x0 + (g_x - 1) * grid.spacing) - qx
        end

        for g_y ∈ g_ymin:g_ymax
            # Check if qy is in this cell, or to above or below
            if qy > grid.y0 + g_y * grid.spacing
                dy = qy - (grid.y0 + g_y * grid.spacing)
            elseif qy >= grid.y0 + (g_y - 1) * grid.spacing
                dy = zero(T)
            else
                dy = (grid.y0 + (g_y - 1) * grid.spacing) - qy
            end

            hdist² = dx*dx + dy*dy
            if hdist² <= r²
                @inbounds push!(grid_cells, grid.grid[g_x, g_y])
                #push!(horizontal_distance², hdist²)
                push!(dz, sqrt(r² - hdist²))
            end
        end
    end

    SpatialIterator(grid_cells, dz, query_point, radius, zmin, zmax)
end

@inline function Base.start(si::SpatialIterator)
    if length(si.grid_cells) == 0
        return (1,0)
    end

    (g, i) = getfirst(si)
    return (g, i)
end

@inline function Base.next(si::SpatialIterator, gi::Tuple{Int, Int})
    (g, i) = getnext(gi[1], gi[2], si)
    @inbounds return (si.grid_cells[gi[1]][gi[2]], (g, i))
end

@inline function Base.done(si::SpatialIterator, gi::Tuple{Int, Int})
    return gi[1] > size(si.grid_cells, 1)
end

@inline function getfirst{T}(si::SpatialIterator{T, Void, Void})
    g = 1
    @inbounds cell = si.grid_cells[g]
    @inbounds dz = si.dz[g]
    # Interesting new (undocumented) interface to sorting in Julia Base:
    # Note that the old method is currently not type stable in Julia 0.5 and
    # results in a noticeable slowdown of the entire algorithm!
    #i = searchsortedfirst(cell, si.query_point - SVector(0.0, 0.0, dz); by = p -> p[3]) # by is applied to comparitor as well... (???)
    i = searchsortedfirst(cell, si.query_point - SVector(0.0, 0.0, dz), Base.By(p -> p[3]))
    r² = si.radius * si.radius

    while true # Loop over g
        while true # Loop over i
            if i > size(cell, 1)
                break
            end

            @inbounds p = cell[i]
            @inbounds if p[3] > si.query_point[3] + dz
                break
            end

            diff = si.query_point - p
            if dot(diff, diff) <= r²
                return (g, i)
            end

            i += 1
        end

        g += 1
        if g > size(si.grid_cells, 1)
            return (g, i)
        else
            @inbounds cell = si.grid_cells[g]
            @inbounds dz = si.dz[g]
            #i = searchsortedfirst(cell, si.query_point - SVector(0.0, 0.0, dz); by = p -> p[3]) # by is applied to comparitor as well... (???)
            i = searchsortedfirst(cell, si.query_point - SVector(0.0, 0.0, dz), Base.By(p -> p[3]))
        end
    end
end


@inline function getnext{T}(g::Int, i::Int, si::SpatialIterator{T, Void, Void})
    @inbounds cell = si.grid_cells[g]
    i += 1
    r² = si.radius * si.radius
    @inbounds dz = si.dz[g]


    while true # Loop over g

        while true # Loop over i
            if i > size(cell, 1)
                break
            end

            @inbounds p = cell[i]
            @inbounds if p[3] > si.query_point[3] + dz
                break
            end

            diff = si.query_point - p
            if dot(diff, diff) <= r²
                return (g, i)
            end

            i += 1
        end

        g += 1
        if g > size(si.grid_cells, 1)
            return (g, i)
        else
            @inbounds cell = si.grid_cells[g]
            @inbounds dz = si.dz[g]
            #i = searchsortedfirst(cell, si.query_point - SVector(0.0, 0.0, dz); by = p -> p[3]) # by is applied to comparitor as well... (???)
            i = searchsortedfirst(cell, si.query_point - SVector(0.0, 0.0, dz), Base.By(p -> p[3]))
        end
    end
end


@inline function height_less(p1::SVector{3}, p2::SVector{3})
    @inbounds z1 = p1[3]
    @inbounds z2 = p2[3]
    return z1 < z2
end


function bounds{T}(points::Vector{SVector{3,T}})
    xmin = typemax(T)
    xmax = typemin(T)
    ymin = typemax(T)
    ymax = typemin(T)

    for p ∈ points
        @inbounds x = p[1]
        if x < xmin
            xmin = x
        end
        if x > xmax
            xmax = x
        end

        @inbounds y = p[2]
        if y < ymin
            ymin = y
        end
        if y > ymax
            ymax = y
        end
    end

    return (xmin, xmax, ymin, ymax)
end

end # module
