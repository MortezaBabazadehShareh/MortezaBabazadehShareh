
using DataFrames
using CSV
using Plots


raw_data=CSV.File("us_data.csv") |> DataFrame
data=Matrix(raw_data)'



### modeling with Sindy ###############################
######################################################

using DataDrivenDiffEq
using ModelingToolkit, OrdinaryDiffEq, LinearAlgebra, Plots
using DataDrivenSparse

### definition of the problem

sir_problem=DiscreteDataDrivenProblem(data)

### definition of polynomial basis
@variables t w(t) x(t) y(t) z(t) 
u = [w;x;y;z]
basis = Basis(polynomial_basis(u, 5), u, iv = t)

### choosing STLSQ as an optimizer
### STLSQ is a sparsifying algorithm that cause the solve function to call its "Sindy" method ###
opt = STLSQ(exp10.(-10:0.01:-1))

### solving the problem to extract the model

ddsol = solve(sir_problem, basis, opt, max_iter=1_0000, options = DataDrivenCommonOptions(digits = 1))

### final differential equations
println(get_basis(ddsol))

### coefficients in the above differential equations
system=get_basis(ddsol);
params=get_parameter_map(system)

plot(plot(sir_problem, title="data"),plot(ddsol, title="model"), layout=(1,2))


