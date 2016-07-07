# tzero implementation from ControlSystems.jl

# Implements the algorithm described in:
# Emami-Naeini, A. and P. Van Dooren, "Computation of Zeros of Linear
# Multivariable Systems," Automatica, 18 (1982), pp. 415–430.
#
# Note that this returns either Vector{Complex64} or Vector{Float64}
function tzero(A::Matrix{Float64}, B::Matrix{Float64}, C::Matrix{Float64},
        D::Matrix{Float64})
    # Balance the system
    A, B, C = balance_statespace(A, B, C)

    # Compute a good tolerance
    meps = 10*eps()*norm([A B; C D])
    A, B, C, D = reduce_sys(A, B, C, D, meps)
    A, B, C, D = reduce_sys(A', C', B', D', meps)
    if isempty(A)   return Float64[]    end

    # Compress cols of [C D] to [0 Df]
    mat = [C D]
    # To ensure type-stability, we have to annote the type here, as qrfact
    # returns many different types.
    W = full(qrfact(mat')[:Q], thin=false)::Matrix{Float64}
    W = flipdim(W,2)
    mat = mat*W
    if fastrank(mat', meps) > 0
        nf = size(A, 1)
        m = size(D, 2)
        Af = ([A B] * W)[1:nf, 1:nf]
        Bf = ([eye(nf) zeros(nf, m)] * W)[1:nf, 1:nf]
        zs = eig(Af, Bf)[1]
    else
        zs = Float64[]
    end
    return zs
end

"""
Implements REDUCE in the Emami-Naeini & Van Dooren paper. Returns transformed
A, B, C, D matrices. These are empty if there are no zeros.
"""
function reduce_sys(A::Matrix{Float64}, B::Matrix{Float64}, C::Matrix{Float64}, D::Matrix{Float64}, meps::Float64)
    Cbar, Dbar = C, D
    if isempty(A)
        return A, B, C, D
    end
    while true
        # Compress rows of D
        U = full(qrfact(D)[:Q], thin=false)::Matrix{Float64}
        D = U'*D
        C = U'*C
        sigma = fastrank(D, meps)
        Cbar = C[1:sigma, :]
        Dbar = D[1:sigma, :]
        Ctilde = C[(1 + sigma):end, :]
        if sigma == size(D, 1)
            break
        end

        # Compress columns of Ctilde
        V = full(qrfact(Ctilde')[:Q], thin=false)::Matrix{Float64}
        V = flipdim(V,2)
        Sj = Ctilde*V
        rho = fastrank(Sj', meps)
        nu = size(Sj, 2) - rho

        if rho == 0
            break
        elseif nu == 0
            # System has no zeros, return empty matrices
            A = B = Cbar = Dbar = Float64[]
            break
        end
        # Update System
        n, m = size(B)
        Vm = [V zeros(n, m); zeros(m, n) eye(m)]
        if sigma > 0
            M = [A B; Cbar Dbar]
            Vs = [V' zeros(n, sigma) ; zeros(sigma, n) eye(sigma)]
        else
            M = [A B]
            Vs = V'
        end
        sigma, rho, nu
        M = Vs * M * Vm
        A = M[1:nu, 1:nu]
        B = M[1:nu, (nu + rho + 1):end]
        C = M[(nu + 1):end, 1:nu]
        D = M[(nu + 1):end,  (nu + rho + 1):end]
    end
    return A, B, Cbar, Dbar
end

# Determine the number of non-zero rows, with meps as a tolerance. For an
# upper-triangular matrix, this is a good proxy for determining the row-rank.
function fastrank(A::Matrix{Float64}, meps::Float64)
    n, m = size(A, 1, 2)
    if n*m == 0     return 0    end
    norms = Array(Float64, n)
    for i = 1:n
        norms[i] = norm(A[i, :])
    end
    mrank = sum(norms .> meps)
    return mrank
end

function balance_statespace{S}(A::Matrix{S}, B::Matrix{S},
        C::Matrix{S}, perm::Bool=false)
    nx = size(A, 1)
    nu = size(B,2)
    ny = size(C,1)

    # Compute the transformation matrix
    mag_A = abs(A)
    mag_B = maximum([abs(B)  zeros(S, nx, 1)], 2)
    mag_C = maximum([abs(C); zeros(S, 1, nx)], 1)
    T = balance_transform(mag_A, mag_B, mag_C, perm)

    # Perform the transformation
    A = T*A/T
    B = T*B
    C = C/T

    return A, B, C, T
end

function balance_transform{R}(A::Matrix{R}, B::Matrix{R}, C::Matrix{R}, perm::Bool=false)
    nx = size(A, 1)
    # Compute a scaling of the system matrix M
    S = diag(balance([A B; C zeros(size(C*B))], false)[1])
    Sx = S[1:nx]
    Sio = S[nx+1]
    # Compute permutation of x (if requested)
    pvec = perm ? balance(A, true)[2] * [1:nx;] : [1:nx;]
    # Compute the transformation matrix
    T = zeros(R, nx, nx)
    T[pvec, :] = Sio * diagm(1./Sx)
    return T
end

function balance(A, perm::Bool=true)
    n = Base.LinAlg.chksquare(A)
    B = copy(A)
    job = perm ? 'B' : 'S'
    ilo, ihi, scaling = LAPACK.gebal!(job, B)

    S = diagm(scaling)
    for j = 1:(ilo-1)   S[j,j] = 1 end
    for j = (ihi+1):n   S[j,j] = 1 end

    P = eye(Int, n)
    if perm
        if ilo > 1
            for j = (ilo-1):-1:1 cswap!(j, round(Int, scaling[j]), P) end
        end
        if ihi < n
            for j = (ihi+1):n    cswap!(j, round(Int, scaling[j]), P) end
        end
    end
    return S, P, B
end

function cswap!{T<:Number}(i::Integer, j::Integer, X::StridedMatrix{T})
    for k = 1:size(X,1)
        X[i, k], X[j, k] = X[j, k], X[i, k]
    end
end
