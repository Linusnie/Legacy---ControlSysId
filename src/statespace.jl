type StateSpaceModel{T<:Real}
    A::Matrix{T}
    B::Vector{T}
    C::Vector{T}
    D::T
    K::Vector{T}
    Re::T
    Ts::Real
    n::Int
    MSE::Real
    fit::Real
    method::Symbol

    function call{T}(::Type{StateSpaceModel}, A::Matrix{T}, B::Vector{T}, C::Vector{T}, D::T, K::Vector{T}, Re::T, Ts::Real, MSE::Real, fit::Real, method::Symbol)
        n = size(A)[1]

        Ts<0 && error("Ts must be nonnegative")

        return new{T}(A, B, C, D, K, Re, Ts, n, MSE, fit, method)
    end
end

#####################################################################
##                      Constructor Functions                      ##
#####################################################################
@doc """`m = n4sid(d, n, i=n+1, h=i)`

Compute a state space model of order `n` based on time-domain iddata `d` using the n4sid method. If an unstable system is returned try increasing the number of samples or tweaking `i` and `h`.""" ->
function n4sid(d::iddataObject, n::Int=-1, i::Int=(n==-1 ? 5 : n+1), h::Int=i)
    y, u = d.y, d.u
    N = length(y)

    # Arrange data as a Hankel matrix with j columns and i+h row (i past, h future)
    j = N - i - h + 1

    # make sure data is siso
    size(y, 2) == 1 ||  error("n4sid only impoemented for siso systems so far")
    # make sure vcat(Uf, Wp, Yf) is a fat matrix
    2(h+i) < j || error("Not enough input data for identification: must have h+i<(N+1)/3")
    # make sure number of lags is larger than the model order
    n < i | n < h && error("There must be at least as many lags as the model order")


    Up = Float64[u[k+t+1] for k = 0:i-1, t = 0:j-1]
    Uf = Float64[u[k+t+1] for k = i:i+h-1, t = 0:j-1]

    Yp = Float64[y[k+t+1] for k = 0:i-1, t = 0:j-1]
    Yf = Float64[y[k+t+1] for k = i:i+h-1, t = 0:j-1]

    Wp = vcat(Up, Yp)
    Wpp = vcat(Up, Uf[1,:], Yp, Yf[1,:])

    #=
    compute Yf/Wp along Uf using LQ-factorization as described in `Subspace Identification Methods`, Pavel Trnka. An equivalent calculation would be:
        Oh = Yf * hcat(Wp', Uf')*inv((vcat(Wp, Uf)*hcat(Wp', Uf')))[:,1:2*i]*Wp
    Might be possible to speed this up by exploiting Hankel structure...
    =#
    F = qrfact(vcat(Uf, Wp, Yf).')
    L = F[:R].'
    L32 = L[end-h+1:end, h+1:h+2i]
    L22 = L[h+1:h+2i, h+1:h+2i]
    Oh = L32 * (L22 \ Wp)

    F1 = qrfact(vcat(Uf[2:end, :], Wpp, Yf[2:end,:]).')
    copy!(L, F1[:R].')
    L32 = L[end-h+2:end, h:h+2i+1]
    L22 = L[h:h+2i+1, h:h+2i+1]
    Oh1 = L32 * (L22 \ Wpp)

    # Calculate SVD and use only information corresponding to the `k` largest singular values
    U, S, V = svd(Oh)
    n==-1 && (n = sum(S .> sqrt(S[1]*S[end])))
    Gam = U[:,1:n] * diagm(sqrt(S[1:n]))

    # Compute estimate of state trajectory (Xhat1 is approximately Xhat time-shifted one step forward)
    Xhat = Gam \ Oh
    Xhat1 = Gam[1:end-1,:] \ Oh1

    # Estimate model parameters
    XU = vcat(Xhat, u[i+1:i+j]')
    XY = vcat(Xhat1, y[i+1:i+j]')
    Theta = (XU' \ XY')'

    Ahat = Theta[1:n,1:n]
    Bhat = collect(Theta[1:n, n+1:end])
    Chat = collect(Theta[n+1:end, 1:n])
    Dhat = Theta[n+1:end, n+1:end][1]

    # Estimate noise parameters
    eps = XY - Theta*XU
    Sigma = 1/(j-(n+1)) * eps * eps'
    Re = Sigma[end,end] # for scalar disturbance (i.e. single output)
    Khat = Sigma[1:end-1, end] / Re

    x = zeros(n)
    y_est = Array{Float64}(N)
    for i=1:N
        y_est[i] = dot(Chat, x) + Dhat*u[i]
        x = Ahat*x + Bhat*u[i] + Khat*(y[i] - y_est[i])
    end

    # determine quality of fit
    MSE = sum((y-y_est).^2)/N
    fit = 100 * (1 - sqrt(N*MSE)/norm(y-mean(y)))

    return StateSpaceModel(Ahat, Bhat, Chat, Dhat, Khat, Re,  d.Ts, MSE, fit, :n4sid)
end

#####################################################################
##                        Display Functions                        ##
#####################################################################

function Base.show(io::IO, m::StateSpaceModel)
    println(io, "Discrete-time State Space model with $(size(m.A,1)) states:\n",
                "\nx(t+1) = Ax(t) + Bu(t) + Ke(t)\n",
                "\ny(t)   = Cx(t) + Du(t) + e(t)\n",
                "\nSampling time: $(m.Ts) seconds\n",
                "Fit: $(m.fit) %, MSE: $(m.MSE), method: $(m.method)")
end
function Base.showall(io::IO, m::StateSpaceModel)
    println(io, "Discrete-time State Space model with $(size(m.A,1)) states:\n",
                "\nx(t+1) = Ax(t) + Bu(t) + Ke(t)\n",
                "\ny(t)   = Cx(t) + Du(t) + e(t)\n",
                "\nA = \n$(m.A)\n",
                "\nB = \n$(m.B)\n",
                "\nC = \n$(m.C)\n",
                "\nD = \n$(m.D)\n",
                "\nK = \n$(m.K)\n",
                "\nRe = \n$(m.Re)\n",
                "\nSampling time: $(m.Ts) seconds\n",
                "Fit: $(m.fit) %, MSE: $(m.MSE), method: $(m.method)")
end

#=  Example:
    A = [-1.5 -0.66 -0.32;
         1  0 0;
         0 0.25 0]
    B = [1,0,0]
    C = [0,1,-0.4]
    D = 0
    K = [2,0,1]


    N = 100
    u = randn(N)
    e = randn(N) / 10
    y = Array{Float64}(N)
    x = zeros(3)

    for i=1:N
        y[i] = dot(C, x) + D*u[i] + e[i]
        x = A*x + B*u[i] + K*e[i]
    end


    d = iddata(y,u)

    m = n4sid(d, 3, 4, 4)
    showall(m)
=#
