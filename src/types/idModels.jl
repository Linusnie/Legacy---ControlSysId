abstract LTIModel

@doc """`MSE, fit = compare(m, d)`

Compare validation data `d` with the ouput predicted by `m`. Returns the mean square error between the predicted and actual output as `MSE = sum(x->x^2, d.y-y_est)`, and the fit value as `fit = 100 * (1 - MSE/(d.y-mean(d.y)))`""" ->
function compare(m::LTIModel, d::iddataObject)
    # compare true system with estimated model (validation)
    y = d.y
    y_est = pred(m, d)
    M = timehorizon(m)

    E = sum(x->x^2, y-y_est)
    fit = 100 * (1 - sqrt(E)/norm(y[M:end]-mean(y[M:end])))
    return E/(length(d)-M), fit
end
