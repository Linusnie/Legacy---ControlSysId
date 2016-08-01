# ControlSysId
System identification toolbox for Julia

**Basic usage**

Identification data is stored as `iddataObject` and is constructed using `d = iddata(y,u)`. The data can then be accessed as  `d.y` and `d.u`. To estimate a model call the appropriate function with `d` and the number of parameters wanted as inputs. For example `m = armax(d, 5, 2, 3)` computes an ARMAX(5,2,3) model (`nk` is set to 1 by default). Type `?armax` in the REPL for more info.  

**Identification methods:**
- `ar`
- `armax`
- `oe` (output error)
- `bj` (box-jenkins)
- `n4sid` (statespace identification)

Currently only `n4sid` supports mimo identification


**Minimal working example:**
```  
# Define the true system 
A = [0.1, 0.7, 0]
B = [0.1, 0.3, 0.2]
C = [0.11, 0.28, 0]
na, nb, nc = 3, 3, 3

# generate input data+noise and simulate output 
N = 1000
u = randn(N) 
e = randn(N) / 10
y = lsim(u, A, B) + lsim(e, A, C)

# create iddataObject for the input/output data
d = iddata(y,u)

# compute model
m = armax(d, na, nb, nc)
showall(m)
```
