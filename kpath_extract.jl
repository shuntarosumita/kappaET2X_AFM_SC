using HDF5
using FLoops

function kpath_extract(m::Mesh, g::Gfunction)
    p::Parameters = m.prmt
    open(p.Logstr, "a") do log
        println(log, "Now extract kpath and kmesh of GF, susceptibility...")
    end

    ### trace of G function
    giωk_tr::Array{Float64, 3} = sum(real.(@view(g.giωk[m.iω0_f, :, :, :, ζ, ζ]) for ζ in 1:p.nwan))
    k_HSP::Vector{Float64}, giωk_tr_HSP::Vector{ComplexF64} = kpath_extractor(
        p, sum(@view(g.giωk[m.iω0_f, :, :, :, ζ, ζ]) for ζ in 1:p.nwan)
    )

    ### largest eigenvalues of susceptibility
    χU_max::Array{Float64, 3} = Array{Float64}(undef, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3)
    @floop for (ik3, ik2, ik1) in my_iter((1:m.prmt.nk3, 1:m.prmt.nk2, 1:m.prmt.nk1))
        χU_max[ik1, ik2, ik3] = maximum(
            real, eigvals(@view(g.χ0iωk[m.iω0_b, ik1, ik2, ik3, :, :]) * m.U_mat)
        )
    end
    _, χU_max_HSP::Vector{Float64} = kpath_extractor(p, χU_max)

    ##### Save results
    h5open(p.plot_savepath, "cw") do file
        # kmap
        haskey(file, "kmap") && delete_object(file, "kmap")
        kmap = create_group(file, "kmap")
        kmap["χU_max"] = χU_max
        kmap["giωk_tr"] = giωk_tr

        # kpath
        haskey(file, "kpath") && delete_object(file, "kpath")
        kpath = create_group(file, "kpath")
        kpath["kvalue"] = k_HSP
        kpath["χU_max_HSP"] = χU_max_HSP
        kpath["giωk_tr_HSP"] = giωk_tr_HSP
    end

    ### electric/magnetic susceptibility
    ops::Vector{Matrix{ComplexF64}} = [
        kron(diagm([1.0, 1.0, 1.0, 1.0]), σ0) ./ (2.0*sqrt(2)),     # electric charge
        kron(diagm([1.0, 1.0, -1.0, -1.0]), σ0) ./ (2.0*sqrt(2)),   # electric quadrupole
        kron(diagm([1.0, 1.0, 1.0, 1.0]), σ3) ./ (2.0*sqrt(2)),     # longitudinal ferromagnetic spin
        kron(diagm([1.0, 1.0, -1.0, -1.0]), σ3) ./ (2.0*sqrt(2)),   # longitudinal antiferromagnetic spin
        kron(diagm([1.0, 1.0, 1.0, 1.0]), σ1 .+ 1.0im.*σ2) ./ 4.0,  # transverse ferromagnetic spin
        kron(diagm([1.0, 1.0, -1.0, -1.0]), σ1 .+ 1.0im.*σ2) ./ 4.0 # transverse antiferromagnetic spin
    ]
    suffixes::Vector{String} = ["ec", "eq", "lfm", "lafm", "tfm", "tafm"]

    for iop in eachindex(ops)
        χ_op::Array{Float64, 3} = real.(
            calc_multipole_susceptibility(m, g, ops[iop])[m.iω0_b, :, :, :]
        )
        _, χ_op_HSP::Vector{Float64} = kpath_extractor(p, χ_op)

        ##### Save results
        h5open(p.plot_savepath, "cw") do file
            write(file, "kmap/χ_"*suffixes[iop], χ_op)
            write(file, "kpath/χ_"*suffixes[iop]*"_HSP", χ_op_HSP)
        end
    end

    open(p.Logstr, "a") do log
        println(log, "Done.")
    end

    k_HSP, χU_max_HSP, giωk_tr_HSP
end

function kpath_extract(m::Mesh, e::Eliashberg)
    p::Parameters = m.prmt
    open(p.Logstr, "a") do log
        println(log, "Now extract kpath and kmesh of gap function...")
    end

    ### gap function
    # sublattice basis
    Δ_sl::Array{ComplexF64, 5} = zeros(ComplexF64, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    for ζ2 in 1:p.nwan, ζ1 in 1:p.nwan
        Δ_sl[:, :, :, ζ1, ζ2] .= sum(
            @view(e.Δiωk[iω, :, :, :, ζ1, ζ2]) for iω in (m.iω0_f-1):m.iω0_f
        ) ./ 2.0
    end

    # band basis: U(k)^† Δ(k) U(-k)^*
    Δ_band::Array{ComplexF64, 5} = zeros(ComplexF64, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        jk1::Int64 = mod(2-ik1, 1:p.nk1)
        jk2::Int64 = mod(2-ik2, 1:p.nk2)
        jk3::Int64 = mod(2-ik3, 1:p.nk3)
        for ζ4 in 1:p.nwan, ζ3 in 1:p.nwan, ζ2 in 1:p.nwan, ζ1 in 1:p.nwan
            Δ_band[ik1, ik2, ik3, ζ1, ζ2] += (
                conj(m.uk[ik1, ik2, ik3, ζ3, ζ1])
                * sum(e.Δiωk[iω, ik1, ik2, ik3, ζ3, ζ4] for iω in (m.iω0_f-1):m.iω0_f) / 2.0
                * conj(m.uk[jk1, jk2, jk3, ζ4, ζ2])
            )
        end
    end

    ### gap from BdG Hamiltonian
    # define normal-state Hamiltonian and order parameter
    hk::Array{ComplexF64, 5} = copy(m.hk)
    for iwan in 1:p.nwan, ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        hk[ik1, ik2, ik3, iwan, iwan] -= m.μ
    end
    hk_inv::Array{ComplexF64, 5} = permutedims(
        reverse(
            circshift(-hk, (-1, -1, -1, 0, 0)),
            dims=(1, 2, 3)
        ),
        (1, 2, 3, 5, 4)
    )
    hk .= circshift(hk, (e.iQ[1], e.iQ[2], e.iQ[3], 0, 0))
    hk_inv .= circshift(hk_inv, (e.iQ[1], e.iQ[2], e.iQ[3], 0, 0))
    Δk::Array{ComplexF64, 5} = sum(@view(e.Δiωk[iω, :, :, :, :, :]) for iω in (m.iω0_f-1):m.iω0_f) ./ 2.0

    # define BdG Hamiltonian
    hk_bdg = Array{ComplexF64, 5}(undef, p.nk1, p.nk2, p.nk3, 2*p.nwan, 2*p.nwan)
    @view(hk_bdg[:, :, :, 1:p.nwan, 1:p.nwan]) .= hk
    @view(hk_bdg[:, :, :, p.nwan+1:2*p.nwan, p.nwan+1:2*p.nwan]) .= hk_inv
    @view(hk_bdg[:, :, :, 1:p.nwan, p.nwan+1:2*p.nwan]) .= Δk
    @view(hk_bdg[:, :, :, p.nwan+1:2*p.nwan, 1:p.nwan]) .= permutedims(
        conj.(Δk), (1, 2, 3, 5, 4)
    )

    # calculate gap
    ek_bdg = Array{Float64, 4}(undef, p.nk1, p.nk2, p.nk3, 2*p.nwan)
    gap = Array{Float64, 3}(undef, p.nk1, p.nk2, p.nk3)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        @views ek_bdg[ik1, ik2, ik3, :] .= eigvals(hk_bdg[ik1, ik2, ik3, :, :])
        gap[ik1, ik2, ik3] = abs(ek_bdg[ik1, ik2, ik3, p.nwan] - ek_bdg[ik1, ik2, ik3, p.nwan+1])
    end

    ##### Save results
    h5open(p.plot_savepath, "cw") do file
        # SC gap
        name = "gap_$(p.SC_type)_$(e.iQ[1])_$(e.iQ[2])_$(e.iQ[3])"
        haskey(file, name) && delete_object(file, name)
        write(file, name, gap)

        # sublattice-based SC order parameter
        name = "Δsl_$(p.SC_type)_$(e.iQ[1])_$(e.iQ[2])_$(e.iQ[3])"
        haskey(file, name) && delete_object(file, name)
        write(file, name, Δ_sl)

        # band-based SC order parameter
        name = "Δband_$(p.SC_type)_$(e.iQ[1])_$(e.iQ[2])_$(e.iQ[3])"
        haskey(file, name) && delete_object(file, name)
        write(file, name, Δ_band)
    end

    open(p.Logstr, "a") do log
        println(log, "Done.")
    end

    nothing
end

"Extracts points of given quantity along HSP k-path Γ->Y->M'->Γ->M->X->Γ."
function kpath_extractor(p::Parameters, quant::Array{T, 3}) where T
    ##### Path extraction
    k_HSP_pos::Vector{Float64} = [0, 1, 2, 2+sqrt(2), 2+2*sqrt(2), 3+2*sqrt(2), 4+2*sqrt(2)] ./ (4+2*sqrt(2))

    ### Γ -> Y
    k_HSP_ΓY::Vector{Float64} = collect(range(k_HSP_pos[1], k_HSP_pos[2], length=p.nk2÷2+1))
    quant_HSP_ΓY::Vector{T} = quant[1, 1:p.nk2÷2+1, 1]

    ### Y -> M'
    k_HSP_YM2::Vector{Float64} = collect(range(k_HSP_pos[2], k_HSP_pos[3], length=p.nk1÷2+1))
    quant_HSP_YM2::Vector{T} = [
        quant[mod(p.nk1-it+2, 1:p.nk1), p.nk2÷2+1, 1] for it in 1:p.nk1÷2+1
    ]

    ### M' -> Γ
    k_HSP_M2Γ::Vector{Float64} = collect(range(k_HSP_pos[3], k_HSP_pos[4], length=p.nk1÷2+1))
    quant_HSP_M2Γ::Vector{T} = [
        quant[mod(p.nk1÷2+it, 1:p.nk1), p.nk2÷2-it+2, 1] for it in 1:p.nk1÷2+1
    ]

    ### Γ -> M
    k_HSP_ΓM::Vector{Float64} = collect(range(k_HSP_pos[4], k_HSP_pos[5], length=p.nk1÷2+1))
    quant_HSP_ΓM::Vector{T} = [
        quant[it, it, 1] for it in 1:p.nk1÷2+1
    ]

    ### M -> X
    k_HSP_MX::Vector{Float64} = collect(range(k_HSP_pos[5], k_HSP_pos[6], length=p.nk2÷2+1))
    quant_HSP_MX::Vector{T} = [
        quant[p.nk1÷2+1, p.nk2÷2-it+2, 1] for it in 1:p.nk2÷2+1
    ]

    ### X -> Γ
    k_HSP_XΓ::Vector{Float64} = collect(range(k_HSP_pos[6], k_HSP_pos[7], length=p.nk1÷2+1))
    quant_HSP_XΓ::Vector{T} = [
        quant[p.nk1÷2-it+2, 1, 1] for it in 1:p.nk1÷2+1
    ]

    # Extract along HSP k-line:
    k_HSP::Vector{Float64} = vcat(
        k_HSP_ΓY, k_HSP_YM2, k_HSP_M2Γ, k_HSP_ΓM, k_HSP_MX, k_HSP_XΓ
    )
    quant_HSP::Vector{T} = vcat(
        quant_HSP_ΓY, quant_HSP_YM2, quant_HSP_M2Γ, quant_HSP_ΓM, quant_HSP_MX, quant_HSP_XΓ
    )

    k_HSP, quant_HSP
end

function calc_multipole_susceptibility(m::Mesh, g::Gfunction, op::Matrix{ComplexF64})
    p::Parameters = m.prmt

    χ_O = zeros(ComplexF64, m.bnω, p.nk1, p.nk2, p.nk3)
    for ζ4 in 1:p.nwan, ζ3 in 1:p.nwan, ζ2 in 1:p.nwan, ζ1 in 1:p.nwan
        ζ12::Int64 = p.nwan * (ζ1-1) + ζ2
        ζ34::Int64 = p.nwan * (ζ3-1) + ζ4
        for iq3 in 1:p.nk3, iq2 in 1:p.nk2, iq1 in 1:p.nk1
            qvec::Vector{Float64} = [
                2(iq1-1)%p.nk1 / p.nk1 - (2(iq1-1)÷p.nk1),
                2(iq2-1)%p.nk2 / p.nk2 - (2(iq2-1)÷p.nk2),
                2(iq3-1)%p.nk3 / p.nk3 - (2(iq3-1)÷p.nk3)
            ]

            # exp(iq.(r3-r1)): phase factor recovering the nonsymmorphic symmetry
            @views χ_O[:, iq1, iq2, iq3] .+= (
                (cispi(dot(qvec, p.pos[ζ3] .- p.pos[ζ1])) * op[ζ1, ζ2] * op[ζ3, ζ4])
                .* g.χiωk[:, iq1, iq2, iq3, ζ12, ζ34]
            )
        end
    end

    maximum(abs, imag.(χ_O)) > 1e-10 && println(
        "!!!!!! Imaginary part of susceptibility is very large: $(maximum(abs, imag.(χ_O))) !!!!!!"
    )

    χ_O
end
