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
