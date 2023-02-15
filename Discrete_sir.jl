using DataDrivenDiffEq
using DataFrames
using CSV
using ModelingToolkit, OrdinaryDiffEq, LinearAlgebra, Plots
using DataDrivenSparse



######## Reading the Data (normal data) ################
raw_data=CSV.File("us_data.csv") |> DataFrame
data=Matrix(raw_data)'
plot(data', alpha=0.7, ls=:dash, lw=2,  label=["Suseptible" "Infected" "Removed" "Death"], color=[:blue :red :green :black])

######## Definition of the Data Driven Problem ################
dt=1
t=collect(0.0:1.0:319.0);
sir_problem=DiscreteDataDrivenProblem(X, t)

### Definition of the Polynomial Basis ######################
@variables t w(t) x(t) y(t) z(t) 
u = [w;x;y;z]
basis = Basis(polynomial_basis(u, 1), u, iv = t)

### Choosing STLSQ as Optimizer #############################
opt = STLSQ(exp10.(-5:0.01:-1))

######## Solving the Problem ################################
ddsol = solve(sir_problem, basis, opt)

### Final Result ##########################################
println(get_basis(ddsol))
plot(plot(sir_problem, title="data"),plot(ddsol, title="model"), layout=(1,2))

##### Recovering the Dynamic and Simulation ################
dudt = let b = get_basis(ddsol)
    (u,p,t) -> b(u,p,t)
end

u0=data[:,1]
tspan=(0.0,320.0)
estimation_prob = DiscreteProblem(dudt, u0, tspan, get_parameter_values(system))
estimate = solve(estimation_prob, FunctionMap(), saveat=dt)
plot(estimate, lw=2, label=["Suseptible" "Infected" "Removed" "Death"], color=[:blue :red :green :black])
plot!(data', alpha=0.7, ls=:dash, lw=2,  label=["Suseptible" "Infected" "Removed" "Death"], color=[:blue :red :green :black])
