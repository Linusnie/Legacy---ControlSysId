function PEM(d::iddataObject, model::Symbol, n::Vector{Int}, x0::AbstractArray, autodiff::Bool = false)
    size(d.y, 2) != 1 && error("PEM only implemented for siso systems")

    k = sum(n[1:end-1]) # number of parameters

    last_x  = Array{Float64}(k)
    last_V  = [-1.]


    if !autodiff
        # select appropriate function to minimize
        if model == :armax
            calc_der! = calc_armax_der!
        elseif model == :oe
            calc_der! = calc_oe_der!
        elseif model === :bj
            calc_der! = calc_bj_der!
        end
        storage = Array{Float64}(k, k+1)

        # Construct TwiceDifferentiableFunction (stores value function, gradient and Hessian). See Optim.jl documentaiton
        df = TwiceDifferentiableFunction(x    -> calc_der!(d, n, x, last_x, last_V, storage),
                                        (x,g) -> g!(calc_der!, d, n, x, last_x, last_V, g, storage),
                                        (x,H) -> h!(calc_der!, d, n, x, last_x, last_V, H, storage))

        return optimize(df, x0)
    else
        # use automatic differentiation (note: this method only computes the gradient, and not the hessian)
        if model == :armax
            calc = calc_armax
        elseif model == :oe
            calc = calc_oe
        elseif model == :bj
            error("automatic differentiation not avaliable for BJ-models")
        end

        # Construct automatic TwiceDifferentiableFunction using autodiff. See Optim/src/autodiff.jl and the documentation for DualNumbers.jl for more info
        df = Optim.autodiff(x -> calc(d, n, x), Float64, length(x0))
        return optimize(df, x0)
    end
end

# helper functions for TwiceDifferentiableFunction
function g!(calc_der!, d, n, x, last_x, last_V, g, storage)
    calc_der!(d, n, x, last_x, last_V, storage)
    copy!(g, storage[:, end])
end
function h!(calc_der!, d, n, x, last_x, last_V, H, storage)
    calc_der!(d, n, x, last_x, last_V, storage)
    copy!(H, storage[:,1:end-1])
end

#=   Function: init_cond

Calculates intitial conditions for PEM.

Author : Cristian Rojas, Linus H-Nielsen
=#
function init_cond(y, u, na, nb, nc, nd=0)
    # TODO: add checks to make sure there are enough samples
    trans = 50
    nf = 3*nc + 6*nd
    nl = 6*nc

    # TODO: allow for time delayed input
    nk = 1

    N = length(u)
    M = max(na, nb+nk-1) + 1

    # compute arx model ignoring noise terms
    theta, phi = arx_(y, u, na, nb, nk)

    # generate instrumental variables
    z = generate_instruments(theta, u, na, nb)[:,M:end-1]

    # compute instrumental variables estimate
    theta = (z*phi) \ z*y[M:end]

    if nd >= 1
        # Obtain residuals v(t) = C/D*e(t) = y(t) - B/A*u(t)
        v   = zeros(N)
        tmp = zeros(na+nb)
        vtmp = zeros(na)
        for i = 1:N
            v[i] = y[i] + dot(tmp, theta) + dot(vtmp, theta[1:na])
            tmp = [y[i]; tmp[1:na-1]; -u[i]; tmp[na+1:na+nb-1]]
            vtmp = [-v[i]; vtmp[1:na-1]]
        end

        # approximate the residuals as a high order AR process
        F = ar_(v, nf)

        # Compute e(t) = v(t) / F
        e   = zeros(N)
        tmp = zeros(nf)

        for i = 1:N
            e[i] = v[i] + dot(tmp, F)
            tmp  = [-e[i]; tmp[1:nf-1]]
        end

        # estimate noise parameters
        theta2 = arx_(v, e, nd, nc, 1)[1]

        # Compute v₂(t) = C/D*e(t)
        tmp = zeros(nc+nd)
        for i = 1:N
            v[i] = e[i] + dot(tmp, theta2)
            tmp = [-v[i]; tmp[1:nd-1]; e[i]; tmp[nd+1:nd+nc-1]]
        end

        theta = arx_(u, y-v, nb, na, 1)[1]
        A = theta[1:na]
        B = theta[na+1:end]
        C = theta2[nd+1:end]
        D = theta2[1:nd]
    else
        # v(t) = C*e(t) = A*y(t) - B*u(t)
        tmp = zeros(na+nb)
        v = zeros(N)
        for i = 1:N
            v[i] = y[i] + dot(tmp, theta)
            tmp  = [y[i]; tmp[1:na-1]; -u[i]; tmp[na+1:na+nb-1]]
        end

        # Estimate L=1/C as a high order AR process
        L = ar_(v, nl)

        # Obtain e(t) = v(t) / C
        e   = zeros(N)
        tmp = zeros(nl)
        for i = 1:N
            e[i] = v[i] + dot(tmp, L)
            tmp  = [-e[i]; tmp[1:nl-1]]
        end

        # Estimate A, B, C from y, u and e
        if nc >= 1
            phi    = zeros(na+nb+nc,1)
            theta  = zeros(na+nb+nc,1)
            P      = 10*var(y)*eye(na+nb+nc)

            for i = trans+1:N
                temp  = phi'*theta
                r     = y[i] - temp[1]
                temp  = 1 + phi'*P*phi
                P     = P - P*phi*phi'*P/temp[1]
                theta = theta + P*phi*r
                phi   = [-y[i]; phi[1:na-1]; u[i]; phi[na+1:na+nb-1]; e[i]; phi[na+nb+1:na+nb+nc-1]]
            end

            A = collect(theta[1:na])
            B = theta[na+1:na+nb]
            C = collect(theta[na+nb+1:na+nb+nc])
            D = Float64[]

        else
            phi    = zeros(na+nb,1)
            theta  = zeros(na+nb,1)
            P      = 10*var(y)*eye(na+nb)
            for i = trans+1:N
                temp  = phi'*theta
                r     = y[i] - temp[1]
                temp  = 1 + phi'*P*phi
                P     = P - P*phi*phi'*P/temp[1]
                theta = theta + P*phi*r
                phi   = [-y[i]; phi[1:na-1]; u[i]; phi[na+1:na+nb-1]]
            end

            A = collect(theta[1:na])
            B = theta[na+1:na+nb]
            C = Float64[]
            D = Float64[]
        end
    end
    return A,B,C,D
end

#=  Generate instrumental variables for use in init_cond
    returns z(t) = (-x(t-1) ... -x(t-na) u(t-1) ... u(t-nb)) where x(t) = A/B*u
    theta = [A;B]
=#
function generate_instruments(theta, u, na, nb)
    N = length(u)
    M = max(na, nb+nb-1) + 1
    # generate instruments
    z = zeros(na+nb, N+1)
    for i=1:N
        z[:,i+1] = [-dot(z[:,i],theta); z[1:na-1,i]; u[i]; z[na+1:na+nb-1,i]]
    end
    return z
end

# models v as an AR process: A*v ≈ e
function ar_(v, nk)
    theta  = zeros(nk,1)
    phi    = zeros(nk,1)
    P      = 10*var(v)*eye(nk)
    for i = 1:length(v)
        e     = v[i] - phi'*theta
        temp  = 1 + phi'*P*phi
        P     = P - P*phi*phi'*P/temp[1]
        theta = theta + P*phi*e
        phi   = [-v[i]; phi[1:nk-1]]
    end
    return collect(theta)
end
