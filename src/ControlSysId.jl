module ControlSysId

export
        # datatypes
        iddataObject,
        LTIModel,
        ArxModel,
        ArmaxModel,
        OeModel,
        BjModel,
        StateSpaceModel,
        iddata,
        # utilities
        detrend,
        detrend!,
        dare,
        compare,
        lsim,
        # identification methods
        arx,
        armax,
        oe,
        bj,
        n4sid,
        pred


using Requires, Polynomials, Optim

import Base.==
import Optim.DualNumbers: Dual, realpart

include("types/iddata.jl")
include("types/idModels.jl")

include("utilities.jl")
include("statespace.jl")
include("PEM.jl")
include("arx.jl")
include("armax.jl")
include("oe.jl")
include("bj.jl")
include("tzero.jl")

end
