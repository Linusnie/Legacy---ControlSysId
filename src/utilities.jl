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
