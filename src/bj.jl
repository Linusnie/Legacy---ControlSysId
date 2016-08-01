import Optim.DualNumbers.realpart
realpart(x) = x

#####################################################################
##                      Data Type Declarations                     ##
#####################################################################

type BjModel{T<:Real} <: LTIModel
    A::Vector{T}
    B::Vector{T}
    C::Vector{T}
    D::Vector{T}
    Ts::Float64
    na::Int
    nb::Int
    nc::Int
    nd::Int
    nk::Int
    MSE::Float64
    fit::Float64
    opt::Optim.OptimizationResults

    function call{T}(::Type{BjModel}, A::Vector{T}, B::Vector{T}, C::Vector{T}, D::Vector{T}, Ts::Float64, nk::Int, MSE::Float64, fit::Float64, opt)
        na = length(A)
        nb = length(B)
        nc = length(C)
        nd = length(D)

        Ts<0 && error("Ts must be nonnegative")

        return new{T}(A, B, C, D, Ts, na, nb, nc, nd, nk, MSE, fit, opt)
    end
end

#####################################################################
##                      Constructor Functions                      ##
#####################################################################
@doc """`m = bj(d, na, nb, nc, nd, nk=1)`

Compute the BJ(`na`,`nb`,`nc`,`nd`,`nk`) model:
    y(t) = z^-`nk`B(z)/A(z)u(t) + C(z)/D(z)e(t)
that minimizes the mean square one-step-ahead prediction error for time-domain iddata `d`. An initial parameter guess can be provided by adding `x0 = [A; B; C; D]` to the argument list, where `A`, `B`, `C` and `D` are vectors. To use automatic differentiation add `autodiff=true`.""" ->

function bj(d::iddataObject, na::Int, nb::Int, nc, nd, nk::Int=1;
                x0::AbstractArray = vcat(init_cond(d.y, d.u, na, nb, nc, nd)...),
                autodiff::Bool = false)
    N = size(d.y, 1)
    M = max(na, nb+nk-1, nc)+1
    n = [na,nb,nc,nd,nk]
    k = na + nb + nc + nd

    # detect input errors
    any(n .< 0)     && error("na, nb, nc, nd, nk must be nonnegative integers")
    M>N             && error("Not enough datapoints to fit BJ($na,$nb,$nc,$nd,$nk) model")
    length(x0) != k && error("Used initial guess of length $(length(x0)) for BJ model with $k parameters")

    opt = PEM(d, :bj, n, x0, autodiff)
    isnan(opt) && (return NaN)

    # extract results from opt
    x = opt.minimum
    MSE = opt.f_minimum
    fit = 100 * (1 - sqrt((N-M)*MSE) / norm(d.y[M:N]-mean(d.y[M:N])))

    A = collect(x[1:na])
    B = collect(x[na+1:na+nb])
    C = collect(x[na+nb+1:na+nb+nc])
    D = collect(x[na+nb+nc+1:end])

    return BjModel(A, B, C, D, d.Ts, nk, MSE, fit, opt)
end

# Calculate the value function V. Used for automatic differentiation
function calc_bj(d, n, x, last_x, last_V)
    # check if this is a new point
    if x != last_x
        # update last_x
        copy!(last_x, realpart(x))

        y, u = d.y, d.u
        na, nb, nc, nd, nk = (n...)
        nca          = nc + na;
        nda          = nd + na;
        ndb          = nd + nk + nb - 1;

        N = length(y)
        M = max(2*na, 2*nb+nk-1, 2*nc, na+nc)+1
        k = na+nb+nc+nd

        A = [1; x[1:na]]
        B = x[na+1:na+nb]
        C = [1; x[na+nb+1:na+nb+nc]]
        D = [1; x[na+nb+nc+1:end]]

        # j = mod(i-1, nc+na) + 1
        # inds = mod((i-1:-1:i-nc-na)-1, nc+na) + 1

        # calculate products of polynomials
        AC = conv(A,C)
        BD = conv(B,D)
        AD = conv(A,D)

        y_est = zeros(N)
        # initiate value function
        V = 0

        # integrate the model and calculate the prediction error
        for i = M:N
            # only the last na+nc values of y_est needs to be stored at any point in time
            # j = modinds(i, nca)

            y_est[i]    += dot(BD, u[i-nk:-1:i-ndb])
            y_est[i]    += dot(AC, y[i:-1:i-nca])
            y_est[i]    -= dot(AD, y[i:-1:i-nda])
            y_est[i]    -= dot(AC[2:end], y_est[i-1:-1:i-nca]);

            # update cost function (prediction error)
            V   += (y[i] - y_est[i])^2
        end

        # normalize
        V /= N-M+1

        # update last_V
        copy!(last_V, V)

        return V
    end
    return last_V[1]
end



# Returns the value function V and saves the gradient/hessian in `storage` as storage = [H g].
function calc_bj_der!(d, n, x, last_x, last_V, storage)
    # check if this is a new point
    if x != last_x
        # update last_x
        copy!(last_x, x)

        y, u = d.y, d.u
        na, nb, nc, nd, nk = (n...)
        nca          = nc + na;
        nda          = nd + na;
        ndb          = nd + nk + nb - 1;

        N = length(y)
        M = max(2*na, 2*nb+nk-1, 2*nc, nca, nda, ndb)+1
        k = na+nb+nc+nd

        A = [1; x[1:na]]
        B = x[na+1:na+nb]
        C = [1; x[na+nb+1:na+nb+nc]]
        D = [1; x[na+nb+nc+1:end]]

        # calculate relevant products of polynomials
        AC = conv(A,C)
        BD = conv(B,D)
        AD = conv(A,D)

        y_est = zeros(N)
        Psi = zeros(k, N) # set all initial conditions on the derivative to zero (not sure if this is correct..)

        # initiate value function
        V = 0
        # initiate gradient and hessian
        copy!(storage, zeros(k, k+1))

        # integrate the model and calculate the prediction error and its derivatives. gradient/Hessian calculations are based on the procedure described in Glad+Ljung modelling and simulation section 12.6
        for i = M:N
            # TODO: make use of moduar indices like in armax etc..

            y_est[i]    += dot(BD, u[i-nk:-1:i-ndb])
            y_est[i]    += dot(AC, y[i:-1:i-nca])
            y_est[i]    -= dot(AD, y[i:-1:i-nda])
            y_est[i]    -= dot(AC[2:end], y_est[i-1:-1:i-nca]);


            # calculate new psi-column
            for k=1:na
                Psi[k,i] = dot(C,y[i-k:-1:i-k-nc]-y_est[i-k:-1:i-k-nc]) - dot(D,y[i-k:-1:i-k-nd])
            end
            for k=1:nb
                Psi[k+na,i] = dot(D,u[i-k:-1:i-k-nd])
            end
            for k=1:nc
                Psi[k+na+nb,i] = dot(A,y[i-k:-1:i-k-na] - y_est[i-k:-1:i-k-na])
            end
            for k=1:nd
                Psi[k+na+nb+nc,i] = dot(B,u[i-nk-k:-1:i-k-nk-nb+1]) - dot(A,y[i-k:-1:i-k-na])
            end
            Psi[:,i] -= Psi[:,i-1:-1:i-nca] * AC[2:end]

            # update cost function (prediction error)
            V   += (y[i] - y_est[i])^2
            # update gradient
            storage[:, end]     -= Psi[:,i]  * (y[i] - y_est[i])
            # update hessian (ignoring second order terms)
            storage[:, 1:end-1] += Psi[:,i] * Psi[:,i].'
        end

        # normalize
        storage[:]  /= N-M+1
        V           /= N-M+1

        # update last_V
        copy!(last_V, V)

        return V
    end
    return last_V[1]
end

#####################################################################
##                        prediction                               ##
#####################################################################

timehorizon(m::BjModel) = max(2*m.na, 2*m.nb+m.nk-1, 2*m.nc, m.na+m.nc) + 1
function pred(m::BjModel, d::iddataObject)
    y, u = d.y, d.u

    N = size(u, 1)
    na, nb, nc, nd, nk = m.na, m.nb, m.nc, m.nd, m.nk
    M = timehorizon(m)
    nca          = nc + na;
    nda          = nd + na;
    ndb          = nd + nk + nb - 1;

    AC = conv([1;m.A],[1;m.C])
    BD = conv(m.B,[1;m.D])
    AD = conv([1;m.A],[1;m.D])

    y_est = zeros(N)

    # initiate value function
    V = 0

    # integrate the model and calculate the prediction error
    for i = M:N
        # only the last na+nc values of y_est needs to be stored at any point in time

        y_est[i]    += dot(BD, u[i-nk:-1:i-ndb])
        y_est[i]    += dot(AC, y[i:-1:i-nca])
        y_est[i]    -= dot(AD, y[i:-1:i-nda])
        y_est[i]    -= dot(AC[2:end], y_est[i-1:-1:i-nca]);
    end
    return y_est
end

#####################################################################
##                        Display Functions                        ##
#####################################################################

modinds(inds, k) = mod(inds-1, k) + 1
function Base.show(io::IO, m::BjModel)
    na, nb, nc, nd, nk = m.na, m.nb, m.nc, m.nd, m.nk
    print(io,   "Discrete-time BJ$((na,nb,nc,nd,nk)) model: y = B(z)/A(z)u + C(z)/D(z)e\n",
                "Sampling time: $(m.Ts) seconds \n",
                "Fit: $(m.fit) %, MSE: $(m.MSE)")
end
function Base.showall(io::IO, m::BjModel)
    A, B, C, D = m.A, m.B, m.C, m.D
    na, nb, nc, nd, nk = m.na, m.nb, m.nc, m.nd, m.nk
    print(io,   "Discrete-time BJ$((na,nb,nc,nd,nk)) model: y = B(z)/A(z)u + C(z)/D(z)e\n",
                "\nA(z) = 1",
                [" $(A[i]<0?(:-):(:+)) $(abs(A[i]))z^-$i" for i=1:na]...,"\n",
                "\nB(z) = $(B[1])$(nk!=0 ? "z^-$nk" : "")",
                [" $(B[i]<0?(:-):(:+)) $(abs(B[i]))z^-$(i+nk-1)" for i=2:nb]...,"\n",
                "\nC(z) = 1",
                [" $(C[i]<0?(:-):(:+)) $(abs(C[i]))z^-$i" for i=1:nc]...,"\n",
                "\nD(z) = 1",
                [" $(D[i]<0?(:-):(:+)) $(abs(D[i]))z^-$i" for i=1:nd]...,"\n",
                "\nSampling time: $(m.Ts) seconds",
                "\nFit: $(m.fit) %, MSE: $(m.MSE)")
end
