#####################################################################
#=                     IDDATA
            Datatype for System Identification

N    :   number of samples
nu   :   number of input channel
ny   :   number of output channel
Ts   :   Sampling time
y    :   N by ny matrix
u    :   N by nu matrix

Author : Lars Lindemann @2015

                                                                   =#
#####################################################################
#=
Do this instead: arx,arma,ar,... <: LTIModel
=#



type iddataObject{T<:Real}
    y::Array{T}
    u::Array{T}
    Ts::Real
    N::Int
    nu::Int
    ny::Int
    inputnames::Vector{UTF8String}
    outputnames::Vector{UTF8String}

    function call{T}(::Type{iddataObject}, y::Array{T}, u::Array{T}, Ts::Real, outputnames, inputnames)
        N   = size(y, 1);
        ny  = size(y, 2);
        nu  = size(u, 2);

        # Validating amount of samples
        if size(y, 1) != size(u, 1)
            error("Input and output need to have the same amount of samples")
        end

        # Validate names of state, input, and output
        if length(inputnames)!=nu
            error("Inputnames need to match the number of channel inputs")
        elseif length(outputnames)!=ny
            error("Outputnames need to match the number of channel outputs")
        end

        # Validate sampling time
        if Ts <= 0
            error("Ts must be a real, positive number")
        end
        new{T}(y, u, Ts, N, nu, ny, inputnames, outputnames)
    end
end

#####################################################################
##                      Constructor Functions                      ##
#####################################################################
@doc """`iddata = iddata(y, u, Ts=1, outputnames="", inputnames="")`

Creates an iddataObject that can be used for System Identification. y and u should have the data arranged in columns.
Use for example sysIdentData = iddata(y1,[u1 u2],Ts,"Out",["In1" "In2"])""" ->
function iddata(y::Array, u::Array, Ts=1; kwargs...)
    nu = size(u,2)
    ny = size(y,2)

    kvs = Dict(kwargs)
    inputnames = validate_names(kvs, :inputnames, nu)
    outputnames = validate_names(kvs, :outputnames, ny)

    y,u = promote(y,u)
    return iddataObject(y, u, Ts, outputnames, inputnames)
end

#####################################################################
##                          Misc. Functions                        ##
#####################################################################
## INDEXING ##
Base.ndims(::iddataObject) = 2
Base.size(d::iddataObject) = (d.ny,d.nu)
Base.size(d::iddataObject,i) = i<=2 ? size(d)[i] : 1
Base.length(d::iddataObject) = size(d.y, 1)

#####################################################################
##                         Math Operators                          ##
#####################################################################

## EQUALITY ##
function ==(d1::iddataObject, d2::iddataObject)
    fields = [:Ts, :inputnames, :outputnames, :u, :y]
    for field in fields
        if getfield(d1,field) != getfield(d2,field)
            return false
        end
    end
    return true
end

#####################################################################
##                        Display Functions                        ##
#####################################################################
Base.print(io::IO, d::iddataObject) = show(io, d)

function Base.show(io::IO, dat::iddataObject)
    println(io, "Discrete-time data set with $(dat.N) samples.")
    println(io, "Sampling time: $(dat.Ts) seconds")
    inputs = format_names(dat.inputnames, "u", "?")
    outputs = format_names(dat.outputnames, "y", "?")
    print(io, "\nOutputs:", ["\n$y" for y in outputs]..., "\n")
    print(io, "\nInputs:", ["\n$u" for u in inputs]...)
end
function Base.showall(io::IO, dat::iddataObject)
    println(io, "Discrete-time data set with $(dat.N) samples.")
    println(io, "Sampling time: $(dat.Ts) seconds")
    outputs = format_names(dat.outputnames, "y", "?")
    inputs = format_names(dat.inputnames, "u", "?")
    print(io, "\nOutputs:", ["\n$(outputs[i]): $(dat.y[:,i])" for i=1:dat.ny]..., "\n")
    print(io, "\nInputs:", ["\n$(inputs[i]): $(dat.u[:,i])" for i=1:dat.nu]...)
end

# Ensures the metadata for an LTISystem is valid
function validate_names(kwargs, key, n)
    names = get(kwargs, key, "")
    if names == ""
        return UTF8String[names for i = 1:n]
    elseif isa(names, Vector) && eltype(names) <: UTF8String
        return names
    elseif isa(names, UTF8String)
        return UTF8String[names * "$i" for i = 1:n]
    else
        error("$key must be of type `UTF8String` or Vector{UTF8String}")
    end
end

# Format the metadata for printing
function format_names(names::Vector{UTF8String}, default::AbstractString, unknown::AbstractString)
    n = size(names, 1)
    if all(names .== "")
        return UTF8String[default * string(i) for i=1:n]
    else
        for (i, n) in enumerate(names)
            names[i] = (n == "") ? unknown : n
        end
        return names
    end
end
