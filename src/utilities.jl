@doc """`d0 = detrend(d)"")`
Returns a copy of iddata ´d´ where the mean has been subtracted from each signal.
""" ->
function detrend(d::iddataObject)
    return iddataObject(d.y.-mean(d.y,1), d.u.-mean(d.u,1), d.Ts, d.outputnames, d.inputnames)
end

@doc """`detrend!(d)"")`
Subtracts the mean from each signal in iddata ´d´.
""" ->
function detrend!(d::iddataObject)
    d.y = d.y.-mean(d.y,1)
    d.u = d.u.-mean(d.u,1)
    return d
end

@doc """`dare(A, B, Q, R, S=0, E=I)`

Compute `X`, the solution to the discrete-time algebraic Riccati equation,
defined as A'XA - E'XE - (A'XB+S)(B'XB + R)^-1(B'XA+S) + Q = 0, where A and R
are non-singular.

Algorithm from:
F. Arnold & J. Laub, "Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations."
""" ->
function dare(A, B, Q, R, S=zeros(size(A,1),size(R,1)), E=eye(size(A,1)))
    n = size(A,1)
    L = [E          B*(R\B');
         zeros(n,n) (A-B*(R\S'))']
    M = [A-B*(R\S')   zeros(n,n);
         S*(R\S')-Q      E']

    F    = schurfact(M,L)
    Ford = ordschur(F, abs(F[:values]) .<= 1)

    W11 = E * Ford[:right][1:n,1:n]
    W21 = Ford[:right][n+1:end,1:n]

    X = (W11' \ W21')'
end


#=   Function: init_cond

Calculates inital start values for the Gauß-Newton Algorithm for
OE and Armax models

Author : Cristian Rojas
                                                                    =#
function init_cond(y, u, na, nb, nc)
    # INIT_COND Determines initial conditions for the parameters of an ARMAX model
    #
    # Input arguments:
    #  u:          Input data (column vector)
    #  y:          Output data (column vector)
    #  na, nb, nc: Degrees of polynomials A, B, C of the model to be estimated
    #
    # Output arguments:
    #  A,B,C:      Vector of initial conditions for the estimated parameters
    #

    trans = 50  # Number of initial data to be removed (due to transients)
    nk    = 20  # Orden of AR polynomial to be estimated

    # Determine data length
    N = length(u)

    # Apply the recursive least squares method
    theta  = zeros(na+nb,1)
    phi    = zeros(na+nb,1)
    P      = 10*var(y)*eye(na+nb)
    for i = 1:N
        e     = y[i] - phi'*theta
        temp  = 1 + phi'*P*phi
        P     = P - P*phi*phi'*P/temp[1]
        theta = theta + P*phi*e
        phi   = [-y[i]; phi[1:na-1]; u[i]; phi[na+1:na+nb-1]]
    end

    # Generate instruments
    z = zeros(na+nb,N+1)
    P = 10*var(y)*eye(na+nb)
    for i = 1:N
        z[:,i+1] = [-z[:,i]'*theta; z[1:na-1,i]; u[i]; z[na+1:na+nb-1,i]]
    end

    # Apply instrumental variables
    phi = zeros(na+nb,1)
    P   = 10*var(y)*eye(na+nb)
    #Th  = zeros(na+nb,0)
    for i = trans+1:N
        e     = y[i] - phi'*theta
        temp  = 1 + phi'*P*z[:,i+1]
        P     = P - P*z[:,i+1]*phi'*P/temp[1]
        theta = theta + P*z[:,i+1]*e
        phi   = [-y[i]; phi[1:na-1]; u[i]; phi[na+1:na+nb-1]]
    end

    # Obtain residuals v(t) = C*e(t)
    v   = zeros(N,1)
    phi = zeros(na+nb,1)
    for i = 1:N
        temp = phi'*theta
        v[i] = y[i] + temp[1]
        phi  = [y[i]; phi[1:na-1]; -u[i]; phi[na+1:na+nb-1]]
    end

    # Apply the recursive least squares method to estimate 1/C as an AR polynomial
    theta  = zeros(nk,1)
    phi    = zeros(nk,1)
    P      = 10*var(y)*eye(nk)
    for i = 1:N
        e     = v[i] - phi'*theta
        temp  = 1 + phi'*P*phi
        P     = P - P*phi*phi'*P/temp[1]
        theta = theta + P*phi*e
        phi   = [-v[i]; phi[1:nk-1]]
    end

    # Obtain e(t) = v(t) / C
    e   = zeros(N,1)
    phi = zeros(nk,1)
    for i = 1:N
        temp = phi'*theta
        e[i] = v[i] + temp[1]
        phi  = [-e[i]; phi[1:nk-1]]
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
        C = []
    end

    return A, B, C
end
