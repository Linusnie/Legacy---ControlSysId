type StateSpaceModel{T<:Real, M1<:AbstractMatrix{T},M2<:AbstractMatrix{T}, M3<:AbstractMatrix{T}, M4<:AbstractMatrix{T}, M5<:AbstractMatrix{T}, M6<:AbstractMatrix{T}}
    A::M1
    B::M2
    C::M3
    D::M4
    K::M5
    Sigma::M6
    Ts::Float64
    n::Int
    MSE::Float64
    fit::Float64
    method::Symbol

    function call{M1<:AbstractMatrix,M2<:AbstractMatrix, M3<:AbstractMatrix,M4<:AbstractMatrix, M5<:AbstractMatrix, M6<:AbstractMatrix}(::Type{StateSpaceModel}, A::M1, B::M2, C::M3, D::M4, K::M5, Sigma::M6, Ts::Float64, MSE::Float64, fit::Float64, method::Symbol)
        n = size(A, 1)

        T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D), eltype(K), eltype(Sigma))

        return new{T,M1,M2,M3,M4,M5,M6}(A, B, C, D, K, Sigma, Ts, n, MSE, fit, method)
    end
end

# overload max() to deal with single entries
Base.max(x) = x

#####################################################################
##                      Constructor Functions                      ##
#####################################################################
@doc """`m = n4sid(d, n, i=n+1, h=i)`

Compute a state space model of order `n` based on time-domain iddata `d` using the n4sid method. If an unstable system is returned try increasing the number of samples or tweaking `i` and `h`.""" ->
function n4sid(d::iddataObject, n::Int=-1, i::Int=(n==-1 ? 5 : n+1), h::Int=i; gamma=0.95)
    y, u = d.y, d.u
    N, ny = size(y,1,2)
    nu = size(u, 2)

    # Arrange data as a Hankel matrix with j columns and i+h row (i past, h future)
    j = N - i - h + 1

    # make sure vcat(Uf, Wp, Yf) is a fat matrix
    2(h+i) < j || error("Not enough input data for identification: must have h+i<(N+1)/3")
    # make sure number of lags is larger than the model order
    n < i | n < h && error("There must be at least as many lags as the model order")

    # Construct Block-Hankel arrays from input/output data
    Up, Uf, Yp, Yf = hankel_data(y, u, i, h)
    Wp = vcat(Up, Yp)
    Wpp = vcat(Up, Uf[1:nu,:], Yp, Yf[1:ny,:])

    #=
    compute the oblique projection Yf/Wp along Uf. An equivalent calculation would be:
    Oh = Yf * hcat(Wp', Uf')*inv((vcat(Wp, Uf)*hcat(Wp', Uf')))[:,1:2*i]*Wp
    Might be possible to speed this up by exploiting Hankel structure...
    =#
    i1, i2, i3 = h*nu, i*(nu+ny), h*ny
    I = i1+i2+i3

    M = Array{Float64}(I, j)
    M[1:i1,:] = Uf
    M[i1+1:i1+i2,:] = Wp
    M[i1+i2+1:end,:] = Yf

    L = Array{Float64}(I, I)
    Oh = obl_proj(M, L, i1, i2, i3)

    # same for Yf-/Wp+ along Uf-
    M[1:(h+i)*nu,:] = M[[nu+1:(h+i)*nu; 1:nu],:]
    Oh1 = obl_proj(M, L, i1-nu, i2+nu+ny, i3-nu)

    # Calculate SVD and use only information corresponding to the `n` largest singular values
    svdinfo = svdfact(Oh)
    S, U = svdinfo[:S], svdinfo[:U]

    n==-1 && (n = sum(S .> sqrt(S[1]*S[end])))
    Gam = U[:,1:n] * diagm(sqrt(S[1:n]))

    # Compute estimate of state trajectory (Xhat1 is approximately Xhat time-shifted one step forward)
    Xhat = Gam \ Oh
    Xhat1 = Gam[1:end-ny,:] \ Oh1

    # Estimate model parameters as Theta = [Ahat Bhat; Chat Dhat]
    XU = vcat(Xhat, u[i+1:i+j,:]')
    XY = vcat(Xhat1, y[i+1:i+j,:]')
    Theta = (XU' \ XY')'

    Ahat = Theta[1:n, 1:n]
    if any(abs(eigfact(Ahat)[:values]) .> 1)
        stabilize!(Theta, Ahat, XU, i, j, ny, nu, n, gamma)
    end

    copy!(Ahat, Theta[1:n,1:n])
    Bhat = Theta[1:n, n+1:end]
    Chat = Theta[n+1:end, 1:n]
    Dhat = Theta[n+1:end, n+1:end]

    # Estimate noise parameters
    Khat, Sigma = noise_param(Theta, Ahat, Chat, XY, XU, n, j)

    # integrate estimated system (replace with lsim later)
    x = zeros(n)
    y_est = Array{Float64}(N,ny)
    for i=1:N
        y_est[i,:] = Chat*x + Dhat*u[i,:]'
        x = Ahat*x + Bhat*u[i,:]' + Khat*(y[i,:] - y_est[i,:])'
    end

    # determine quality of fit
    E = sum(x->x^2, y-y_est)
    MSE = E/N
    fit = 100 * (1 - sqrt(E)/norm(y .- mean(y,1)))

    return StateSpaceModel(Ahat, Bhat, Chat, Dhat, Khat, Sigma, d.Ts, MSE, fit, :n4sid)
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

function hankel_data(y, u, i, h)
    N, ny = size(y,1,2)
    nu = size(u, 2)
    j = N - i - h + 1

    Up = Array{Float64}(i*nu, j)
    for k = 0:i-1, iu = 1:nu, t = 0:j-1
        Up[nu*k+iu, t+1] = u[k+t+1, iu]
    end

    Uf = Array{Float64}(h*nu, j)
    for k = 0:h-1, iu = 1:nu, t = 0:j-1
        Uf[nu*k+iu, t+1] = u[i+k+t+1, iu]
    end

    Yp = Array{Float64}(i*ny, j)
    for k = 0:i-1, iy = 1:ny, t = 0:j-1
        Yp[ny*k+iy, t+1] = y[k+t+1, iy]
    end

    Yf = Array{Float64}(h*ny, j)
    for k = 0:h-1, iy = 1:ny, t = 0:j-1
        Yf[ny*k+iy, t+1] = y[i+k+t+1, iy]
    end
    return Up, Uf, Yp, Yf
end

#=
compute the oblique projection A/C along B using LQ-factorization as described in `Subspace Identification Methods`, Pavel Trnka. L has to be initialized outside the function

notation: M = [B;C;A]
          j,k and l are the number of rows in B,C and A respectively
=#
function obl_proj(M, L, j, k, l)
    F = qrfact(M.')
    copy!(L, F[:R].')
    L32 = L[j+k+1:end, j+1:j+k]
    L22 = L[j+1:j+k, j+1:j+k]
    return L32 * (L22 \ M[j+1:j+k,:])
end

#=
Impose stability on the estimated system by adding a regularization term. This is equivalent to minimizing ||AX - BU||^2 + c||A||^2 (in Forbenius norm). `c` is picked so that the spectral radius of `Ahat` is exactly `gamma` (i.e. the modified estimate is guaranteed to be stable for `gamma` < 1).

See Gestel, Suykens, Dooren, Moor `Imposing Stability in Subspace Identification by Regularization`.
=#
function stabilize!(Theta, Ahat, XU, i, j, ny, nu, n, gamma)
    UX = XU[[n+1:end; 1:n],:]'

    F = qrfact(UX)
    R22 = F[:R][nu+1:end, nu+1:end]
    S = R22' * R22

    P2 = - gamma^2 * eye(n^2)
    P1 = - gamma^2 * kron(eye(n), S) - gamma^2 * kron(S, eye(n))
    P0 = kron(Ahat*S, Ahat*S) - gamma^2 * kron(S, S)

    # solve generalized eigenvalue problem and find the largest real+positive eigenvalue
    theta = eigfact([zeros(n^2,n^2) -eye(n^2); P0 P1], -[eye(n^2) zeros(n^2,n^2); zeros(n^2, n^2) P2])[:values]
    c = max(abs(theta[(imag(theta) .== 0 ) .* (real(theta) .> 0)])...)

    # multiply [Ahat Bhat] with regularization term
    S_XU = XU * XU'
    Theta[1:n,:] *= S_XU * inv(S_XU + c*[eye(n) zeros(n,nu); zeros(nu, n+nu)])
end

# Estimate nosie parameters from state estimate and system model
function noise_param(Theta, A, C, XY, XU, n, j)
    eps = XY - Theta*XU
    Sigma = 1/(j-(n+1)) * eps * eps'
    Q = Sigma[1:n, 1:n]
    R = Sigma[n+1:end, n+1:end]
    S = Sigma[1:n, n+1:end]

    # calculate Kalman gain
    P = dare(A', C', Q, R, S)
    K = ((C*P*C' + R)\(A*P*C' + S)')'
    return K, Sigma
end
