include("parameters.jl")
include("mesh.jl")

function main()
    length(ARGS) < 9 && error("usage: julia fs_renormalized.jl system h n U T nk1 nk2 nk3 wmax")
    system::String = ARGS[1]
    h::Float64 = parse(Float64, ARGS[2])
    n_fill::Float64 = parse(Float64, ARGS[3])
    U::Float64 = parse(Float64, ARGS[4])
    T::Float64 = parse(Float64, ARGS[5])
    nk1::Int64 = parse(Int64, ARGS[6])
    nk2::Int64 = parse(Int64, ARGS[7])
    nk3::Int64 = parse(Int64, ARGS[8])
    wmax::Float64 = parse(Float64, ARGS[9])

    ### Initiate parameters -------------------------------------------------------
    p = Parameters(
        system, "skip", h, n_fill, U, T, 1, "skip"; h_load=h, U_load=U,
        nk1, nk2, nk3, wmax
    )

    ### Load mesh -----------------------------------------------------------------
    m = Mesh(p)
    ek0_u::Array{Float64, 4} = permutedims(
        m.ek0[:, :, :, 1:p.nspin:p.nwan],
        (4, 1, 2, 3)
    )
    ek0_d::Array{Float64, 4} = permutedims(
        m.ek0[:, :, :, 2:p.nspin:p.nwan],
        (4, 1, 2, 3)
    )

    ### Get data ------------------------------------------------------------------
    ## load self-energy and chemical potential
    loadfile = h5open(p.loadpath, "r")
    Σiωk::Array{ComplexF64, 6} = read(loadfile, "gfunction/Σiωk")
    μ::Float64 = read(loadfile, "gfunction/μ")

    ## calculate energy eigenvalues for Hamiltonian + self-energy
    hkrenorm::Array{ComplexF64, 5} = m.hk .+ sum(
        @view(Σiωk[iω, :, :, :, :, :]) for iω in (m.iω0_f-1):m.iω0_f
    ) ./ 2.0
    println(maximum(abs, @view(hkrenorm[:, :, :, 1:p.nspin:p.nwan, 2:p.nspin:p.nwan])))
    ek_u::Array{Float64, 4} = zeros(Float64, p.norb, p.nk1, p.nk2, p.nk3)
    ek_d::Array{Float64, 4} = zeros(Float64, p.norb, p.nk1, p.nk2, p.nk3)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        ek_u[:, ik1, ik2, ik3] .= real.(eigvals(
            @view(hkrenorm[ik1, ik2, ik3, 1:p.nspin:p.nwan, 1:p.nspin:p.nwan])
        ))
        ek_d[:, ik1, ik2, ik3] .= real.(eigvals(
            @view(hkrenorm[ik1, ik2, ik3, 2:p.nspin:p.nwan, 2:p.nspin:p.nwan])
        ))
    end

    ## set wave number k
    k1s::Vector{Float64} = [(2π*(ik1-1)) / p.nk1 for ik1 in 1:p.nk1]
    k2s::Vector{Float64} = [(2π*(ik2-1)) / p.nk2 for ik2 in 1:p.nk2]
    k3s::Vector{Float64} = [(2π*(ik3-1)) / p.nk3 for ik3 in 1:p.nk3]

    ### output data --------------------------------------------------------------
    filename::String = Printf.format(
        Printf.Format("Odata_k-$(system)/specN_h_%.3f_n_%.3f_U_%.3f_T_%.4f.h5"),
        h, n_fill, U, T
    )
    h5open(filename, "w") do file
        write(file, "ek0_u", ek0_u)
        write(file, "ek0_d", ek0_d)
        write(file, "ek_u", ek_u)
        write(file, "ek_d", ek_d)
        write(file, "μ0", m.μ)
        write(file, "μ", μ)
        write(file, "k1s", k1s)
        write(file, "k2s", k2s)
        write(file, "k3s", k3s)
    end

    return nothing
end

main()
