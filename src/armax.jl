#####################################################################
##                      Data Type Declarations                     ##
#####################################################################

type ArmaxModel{T<:Real} <: LTIModel
    A::Vector{T}
    B::Vector{T}
    C::Vector{T}
    Ts::Real
    na::Int
    nb::Int
    nc::Int
    nk::Int
    MSE::Real
    fit::Real
    opt::Optim.OptimizationResults

    function call{T}(::Type{ArmaxModel}, A::Vector{T}, B::Vector{T}, C::Vector{T}, Ts::Real, nk::Int, MSE::Real, fit::Real, opt)
        na = length(A)
        nb = length(B)
        nc = length(C)

        Ts<0 && error("Ts must be nonnegative")

        return new{T}(A, B, C, Ts, na, nb, nc, nk, MSE, fit, opt)
    end
end

#####################################################################
##                      Constructor Functions                      ##
#####################################################################
@doc """`m = armax(d, na, nb, nc, nk=1)`

Compute the ARMAX(`na`,`nb`,`nc`,`nd`) model:
    A(z)y(t) = z^-`nk`B(z)u(t) + C(z)e(t)
that minimizes the mean square one-step-ahead prediction error for time-domain iddata `d`. An initial parameter guess can be provided by adding `x0 = [A; B; C]` to the argument list, where `A`, `B` and `C` are vectors. To use automatic differentiation add `autodiff=true`.""" ->

function armax(d::iddataObject, na::Int, nb::Int, nc, nk::Int=1;
                x0::AbstractArray = vcat(init_cond(d.y, d.u, na, nb, nc)...),
                autodiff::Bool = false)
    N = size(d.y, 1)
    M = max(na, nb+nk-1, nc)+1
    n = [na,nb,nc,nk]
    k = na + nb + nc

    # detect input errors
    any(n .< 0)     && error("na, nb, nc, nk must be nonnegative integers")
    M>N             && error("Not enough datapoints to fit ARMAX($na,$nb,$nc,$nk) model")
    length(x0) != k && error("Used initial guess of length $(length(x0)) for ARMAX model with $k parameters")

    opt = PEM(d, :armax, n, x0, autodiff)

    # extract results from opt
    x = opt.minimum
    MSE = opt.f_minimum
    fit = 100 * (1 - sqrt((N-M)*MSE) / norm(d.y[M:N]-mean(d.y[M:N])))

    A = x[1:na]
    B = x[na+1:na+nb]
    C = x[na+nb+1:end]

    return ArmaxModel(x[1:na], x[na+1:na+nb], x[na+nb+1:end], d.Ts, nk, MSE, fit, opt)
end

# Calculate the value function V. Used for automatic differentiation
function calc_armax(d, n, x)
    y, u = d.y, d.u
    na, nb, nc, nk = (n...)

    A = x[1:na]
    B = x[na+1:na+nb]
    C = x[na+nb+1:end]

    N = length(y)
    M = max(na, nb+nk-1, nc)+1
    k = na+nb+nc

    y_est = zeros(eltype(x), nc)

    # initiate value function
    V = 0

    # integrate the model and calculate the prediction error
    for i = M:N
        # only the last nc values of Phi and y_est needs to be stored at any point in time
        j = mod(i-1, nc) + 1
        inds = mod((i-1:-1:i-nc)-1, nc) + 1

        # calculate one step ahead prediction
        y_est[j] =  - dot(A, y[i-1:-1:i-na]) + dot(B, u[i-nk:-1:i-nb-nk+1]) + dot(C, y[i-1:-1:i-nc] - y_est[inds])

        # update cost function (prediction error)
        V   += (y[i] - y_est[j])^2
    end

    # normalize
    V /= N-M+1

    # update last_V

    return V
end


# Returns the value function V and saves the gradient/hessian in `storage` as storage = [H g].
function calc_armax_der!(d, n, x, last_x, last_V, storage)
    # check if this is a new point
    if x != last_x
        # update last_x
        copy!(last_x, x)

        y, u = d.y, d.u
        na, nb, nc, nk = (n...)

        A = x[1:na]
        B = x[na+1:na+nb]
        C = x[na+nb+1:end]

        N = length(y)
        M = max(na, nb+nk-1, nc)+1
        k = na+nb+nc

        y_est = zeros(nc)
        Psi = zeros(k, nc) # set all initial conditions on the derivative to zero (not sure if this is correct..)

        # initiate value function
        V = 0
        # initiate gradient and hessian
        copy!(storage, zeros(k, k+1))

        # integrate the model and calculate the prediction error and its derivatives. gradient/Hessian calculations are based on the procedure described in Glad+Ljung modelling and simulation section 12.6
        for i = M:N
            # only the last nc values of Phi/y_est needs to be stored at any point in time
            j = mod(i-1, nc) + 1
            inds = mod((i-1:-1:i-nc)-1, nc) + 1

            # calculate new psi-column
            Psi[:, j] = [-y[i-1:-1:i-na];
                        u[i-nk:-1:i-nk-nb+1];
                        y[i-1:-1:i-nc] - y_est[inds]] - Psi[:,inds]*C

            # calculate one step ahead prediction
            y_est[j] =  - dot(A, y[i-1:-1:i-na]) + dot(B, u[i-nk:-1:i-nb-nk+1]) + dot(C, y[i-1:-1:i-nc] - y_est[inds])

            # update cost function (prediction error)
            V   += (y[i] - y_est[j])^2
            # update gradient
            storage[:, end]     -= Psi[:,j]  * (y[i] - y_est[j])
            # update hessian (ignoring second order terms)
            storage[:, 1:end-1] += Psi[:,j] * Psi[:,j].'
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

timehorizon(m::ArmaxModel) = max(m.na, m.nb+m.nk-1, m.nc) + 1
function pred(m::ArmaxModel, d::iddataObject)
    na, nb, nc, nk = m.na, m.nb, m.nc, m.nk
    y, u = d.y, d.u
    N = size(y,1)
    M = timehorizon(m)

    y_est = zeros(N)
    for i = M:N
        y_est[i] =  - dot(m.A, y[i-1:-1:i-na]) + dot(m.B, u[i-nk:-1:i-nb-nk+1]) + dot(m.C, y[i-1:-1:i-nc] - y_est[i-1:-1:i-nc])
    end
    return y_est
end

#####################################################################
##                        Display Functions                        ##
#####################################################################

function Base.show(io::IO, m::ArmaxModel)
    print(io,   "Discrete-time ARMAX$((m.na,m.nb,m.nc,m.nk)) model: A(z)y = B(z)u + C(z)e\n",
                "Sampling time: $(m.Ts) seconds \n",
                "Fit: $(m.fit) %, MSE: $(m.MSE)")
end
function Base.showall(io::IO, m::ArmaxModel)
    A, B, C = m.A, m.B, m.C
    na, nb, nc, nk = m.na, m.nb, m.nc, m.nk
    print(io,   "Discrete-time ARMAX$((na,nb,nc,nk)) model: A(z)y = B(z)u + C(z)e\n",
                "\nA(z) = 1",
                [" $(A[i]<0?(:-):(:+)) $(abs(A[i]))z^-$i" for i=1:na]...,"\n",
                "\nB(z) = $(B[1])$(nk!=0 ? "z^-$nk" : "")",
                [" $(B[i]<0?(:-):(:+)) $(abs(B[i]))z^-$(i+nk-1)" for i=2:nb]...,"\n",
                "\nC(z) = 1",
                [" $(C[i]<0?(:-):(:+)) $(abs(C[i]))z^-$i" for i=1:nc]...,"\n",
                "\nSampling time: $(m.Ts) seconds",
                "\nFit: $(m.fit) %, MSE: $(m.MSE)")
end
