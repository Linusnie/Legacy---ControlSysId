#=   ARX-Model - recursive implementation
For the arx model we assume a structure like
y(n)+a_1*y(n-1)+...+a_na*y(n-na) = b1*u(n-nk)+...+b_nb*u(n-nk-nd+1)+e(n),
which results in
theta = [a1        a2  ...  a_na    b_1     ... b_nb]'
phi   = [-y(n-1)] -y(n-2)   -y(n-na) u(n-nk)     u(n-nk-nb+1)]'

Our best prediction is now:
y(n) = theta^T*phi
                                                                    =#

type ArxModel{T<:Real}
    A::Vector{T}
    B::Vector{T}
    Ts::Real
    na::Int
    nb::Int
    nk::Int
    MSE::Real
    fit::Real

    function call{T}(::Type{ArxModel}, A::Vector{T}, B::Vector{T}, Ts::Real, nk::Int, MSE::Real, fit::Real)
        na = length(A)
        nb = length(B)

        Ts<0 && error("Ts must be nonnegative")

        return new{T}(A, B, Ts, na, nb, nk, MSE, fit)
    end
end

function arx(d::iddataObject, na::Int, nb::Int, nk = 0)
    y, u = d.y, d.u

    size(y,2) != 1 | size(u,2) != 1 && error("arx only implemented for siso")
    na < 1 | nb < 1 | nk < 0 && error("arx parameters must have na,nb > 0 and nk >= 0")

    # Number of samples
    N = length(y)

    # Time horizon
    M = max(na, nb+nk-1)+1

    # Number of parameters
    n = na + nb

    # Estimate parameters
    Phi = Matrix{Float64}(N-M+1, n)
    for i=M:N
        Phi[i-M+1,:] = hcat(-y[i-1:-1:i-na]', u[i-nk:-1:i-nk-nb+1]')
    end
    theta = Phi\y[M:N]
    A = theta[1:na]
    B = theta[na+1:end]

    # Calculate model error
    MSE = norm(y[M:N]-Phi*theta)
    fit = 100*(1 - MSE/norm(y[M:N]-mean(y[M:N])))

    return ArxModel(A, B, d.Ts, nk, MSE/(N-M), fit)
end

function Base.show(io::IO, m::ArxModel)
    print(io, "Discrete-time ARX$((m.na,m.nb,m.nk)) model: A(z)y = B(z)u + e\n",
              "Sampling time: $(m.Ts) seconds\n",
              "Fit: $(m.fit) %, MSE: $(m.MSE)")
end
function Base.showall(io::IO, m::ArxModel)
    A, B = m.A, m.B
    na, nb, nk = m.na, m.nb, m.nk
    print(io, "Discrete-time ARX$((na,nb,nk)) model: A(z)y = B(z)u + e\n",
              "\nA(z) = 1",
              [" $(A[i]<0?(:-):(:+)) $(abs(A[i]))z^-$i" for i=1:na]...,
              "\nB(z) = $(B[1])$(nk!=0 ? "z^-$nk" : "")",
              [" $(B[i]<0?(:-):(:+)) $(abs(B[i]))z^-$(i+nk-1)" for i=2:nb]...,"\n",
              "\nSampling time: $(m.Ts) seconds",
              "\nFit: $(m.fit) %, MSE: $(m.MSE)")
end
