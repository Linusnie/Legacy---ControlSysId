module ControlSysId

export
        # datatypes
        iddataObject,
        LTIModel,
        ArxModel,
        ArmaxModel,
        StateSpaceModel,
        iddata,
        # utilities
        detrend,
        detrend!,
        dare,
        # identification methods
        arx,
        armax,
        n4sid


using Requires, Polynomials, Optim

import Base.==

include("types/iddata.jl")
include("types/idModels.jl")

include("utilities.jl")
include("statespace.jl")
include("arx.jl")
include("armax.jl")
include("tzero.jl")

end
