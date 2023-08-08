using TensorCore
using Printf
using HDF5

const σ0 = ComplexF64[1 0; 0 1]
const σ1 = ComplexF64[0 1; 1 0]
const σ2 = ComplexF64[0 -1im; 1im 0]
const σ3 = ComplexF64[1 0; 0 -1]

##############################################################################
##### Main code: Calculate lin. eliashberg eq. within FLEX approximation #####
##############################################################################
mutable struct Eliashberg
    Viτr::Array{ComplexF64, 6}
    V_DC::Array{ComplexF64, 2}
    Δiωk::Array{ComplexF64, 6}
    fiτr::Array{ComplexF64, 6}
    fiτr_0::Array{ComplexF64, 2}
    iQ::Vector{Int64}
    λ::Float64
end

struct PointGroup
    grp::String
    order::Int64
    irrep::String
    character::Vector{Float64}
end

function Eliashberg(m::Mesh, g::Gfunction, iQ::Vector{Int64})::Eliashberg
    p::Parameters = m.prmt

    Viτr = Array{ComplexF64, 6}(undef, m.bnτ, p.nk1, p.nk2, p.nk3, p.nwan^2, p.nwan^2)
    V_DC = Array{ComplexF64, 2}(undef, p.nwan^2, p.nwan^2)
    Δiωk = Array{ComplexF64, 6}(undef, m.fnω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    fiτr = Array{ComplexF64, 6}(undef, m.fnτ, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    fiτr_0 = Array{ComplexF64, 2}(undef, p.nwan, p.nwan)

    e = Eliashberg(Viτr, V_DC, Δiωk, fiτr, fiτr_0, iQ, 0.0)

    set_Viτr!(m, g, e)

    return e
end

function save_Eliashberg(m::Mesh, g::Gfunction, e::Eliashberg)
    ##### Save results
    open(m.prmt.Logstr, "a") do log
        println(log, "Saving all data now...")
    end

    open(m.prmt.SC_EV_path, "a") do file
        println(file, "$(m.prmt.T) $(e.iQ[1]) $(e.iQ[2]) $(e.iQ[3]) $(e.λ)")
    end

    if e.iQ == [0, 0, 0]
        h5open(m.prmt.savepath, "cw") do file
            name = "eliashberg_$(m.prmt.SC_type)_$(e.iQ[1])_$(e.iQ[2])_$(e.iQ[3])"
            haskey(file, name) && delete_object(file, name)
            group = create_group(file, name)
            group["gap"] = e.Δiωk
            group["λ"] = e.λ
        end
    end
end

##############
### Self consistency loop for linearized Eliashberg equation
### Employs power iterative method to solve λ*Δ = λ*V*F in (τ, r)-space
##############
"""
Self consistency loop for super conduction parameter via eigenvalue method.
Implements FLEX approximation in linearized Eliashberg equation.
Handles depending on SC-type input in p.SC_type(=parameters) the equation differently.
"""
function solve_Eliashberg!(m::Mesh, g::Gfunction, e::Eliashberg)
    set_Δiωk!(m, g, e)
    e.λ = SC_sfc!(m, g, e, 0.0)

    ishift::Int = 1
    while e.λ < 0
        ishift > 10 && break

        open(m.prmt.Logstr, "a") do log
            println(log, "ishift = $ishift: Another eliashberg cycle will be performed.")
        end
        open(m.prmt.Logerrstr, "a") do logerr
            println(logerr, m.prmt.err_str_begin * "ishift = $ishift: λ < 0 => new round!")
        end

        open(m.prmt.SC_EV_path_neg, "a") do file
            println(file, "$(m.prmt.T) $(e.iQ[1]) $(e.iQ[2]) $(e.iQ[3]) $ishift $(e.λ)")
        end

        set_Δiωk!(m, g, e)
        e.λ = SC_sfc!(m, g, e, e.λ)
        ishift += 1
    end
end

function SC_sfc!(m::Mesh, g::Gfunction, e::Eliashberg, λ_in::Float64)
    p::Parameters = m.prmt
    nall::Int = m.fnω * p.nk * p.nwan^2

    λ::ComplexF64 = 0.0
    Δmax::Float64 = 10.0; λmax::Float64 = 10.0
    yiωk::Array{ComplexF64, 6} = zeros(ComplexF64, m.fnω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    pg::PointGroup = PointGroup(m)
    count::Int = 1
    while Δmax > 100*p.SC_sfc_tol || λmax > p.SC_sfc_tol || abs(imag(λ)) > p.SC_sfc_tol * abs(real(λ))
        # save previous λ and Δiωk
        λ_old::Float64 = real(λ)
        Δiωk_old::Array{ComplexF64, 6} = copy(e.Δiωk)

        # Power iteration method for computing λ
        set_fiτr!(m, g, e)

        # y = V*F
        yiτr = zeros(ComplexF64, m.fnτ, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
        for ζ4 in 1:p.nwan, ζ3 in 1:p.nwan, ζ2 in 1:p.nwan, ζ1 in 1:p.nwan
            ζ12::Int64 = p.nwan * (ζ1-1) + ζ2
            ζ34::Int64 = p.nwan * (ζ3-1) + ζ4
            yiτr[:, :, :, :, ζ1, ζ4] .+= @views(
                e.Viτr[:, :, :, :, ζ12, ζ34] .* e.fiτr[:, :, :, :, ζ2, ζ3]
            )
        end

        # Fourier transform
        yiτk::Array{ComplexF64, 6} = r_to_k(m, yiτr)
        yiωk .= τ_to_ωn(m, Fermionic(), yiτk)

        # y_HF
        for ζ4 in 1:p.nwan, ζ3 in 1:p.nwan, ζ2 in 1:p.nwan, ζ1 in 1:p.nwan
            ζ12::Int64 = p.nwan * (ζ1-1) + ζ2
            ζ34::Int64 = p.nwan * (ζ3-1) + ζ4
            yiωk[:, :, :, :, ζ1, ζ4] .+= e.V_DC[ζ12, ζ34] * e.fiτr_0[ζ2, ζ3] / p.nk
        end

        ### y - λ*y (power iteration method trick)
        yiωk .-= λ_in .* e.Δiωk

        ### Impose symmetry conditions
        symmetrize_gap!(m, e.iQ, pg, yiωk)

        ### Calculate λ
        λ = sum(conj(e.Δiωk) .* yiωk) / nall + λ_in
        λmax = abs(λ - λ_old)

        ### Calculate Δiωk
        e.Δiωk .= normalize(yiωk) .* sqrt(nall)
        Δmax = maximum(abs.(e.Δiωk .- Δiωk_old))

        # break a loop if λ converges to a negative value
        (real(λ) < 0.0 && λmax < 1e-4) && break

        (count > 2000) && break

        if count % 50 == 1
            open(p.Logstr, "a") do log
                println(log, "$count: $λ, $λmax, $Δmax")
            end
        end

        count += 1
    end

    open(p.Logstr, "a") do log
        println(log, "$count: $λ, $λmax, $Δmax; loop finished")
    end

    # finilize Δiωk
    maxΔid = findmax(abs, e.Δiωk)[2]
    e.Δiωk ./= e.Δiωk[maxΔid]

    real(λ)
end

function symmetrize_gap!(m::Mesh, iQ::Vector{Int64}, pg::PointGroup, yiωk::Array{ComplexF64, 6})
    # Even function of matsubara frequency
    # yiωk_inv corresponds to y(k, -iωn)
    yiωk_inv::Array{ComplexF64, 6} = reverse(yiωk, dims=1)
    yiωk .= (yiωk .+ yiωk_inv) ./ 2.0

    # Pauli exclusion priciple: y(k)_ab = -y(-k+Q)_ba
    # yiωk_inv corresponds to y(-k+Q, iωn)^T
    yiωk_inv .= reverse(
        circshift(yiωk, (0, -1, -1, -1, 0, 0)),
        dims=(2, 3, 4)
    )
    yiωk_inv .= permutedims(
        circshift(
            yiωk_inv,
            (0, iQ[1], iQ[2], iQ[3], 0, 0)
        ),
        (1, 2, 3, 4, 6, 5)
    )
    if m.prmt.nspin == 2 || (m.prmt.nspin == 1 && m.prmt.SC_type[end] == 't')
        yiωk .= (yiωk .- yiωk_inv) ./ 2.0
    elseif m.prmt.nspin == 1 && m.prmt.SC_type[end] == 's'
        yiωk .= (yiωk .+ yiωk_inv) ./ 2.0
    end

    # point group symmetry (not applicable to FFLO)
    iQ == [0, 0, 0] && (yiωk .= projection_operator(m, pg, yiωk))

    yiωk
end

### Set Coulomb interaction V_a(τ, r) --------------------------------
function set_Viτr!(m::Mesh, g::Gfunction, e::Eliashberg)
    # Set V
    Viωk = Array{ComplexF64, 6}(undef, m.bnω, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3, m.prmt.nwan^2, m.prmt.nwan^2)
    if m.prmt.nspin == 2
        for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.bnω
            Viωk[iω, ik1, ik2, ik3, :, :] .= @views(
                -m.U_mat * g.χiωk[iω, ik1, ik2, ik3, :, :] * m.U_mat
            )
        end
        e.V_DC .= -m.U_mat ./ 2.0

    elseif m.prmt.nspin == 1
        # charge susceptibility
        id = Matrix{ComplexF64}(I, m.prmt.nwan^2, m.prmt.nwan^2)
        denom = copy(id)
        χciωk = Array{ComplexF64, 6}(undef, m.bnω, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3, m.prmt.nwan^2, m.prmt.nwan^2)
        for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.bnω
            denom .= id
            @views mul!(denom, g.χ0iωk[iω, ik1, ik2, ik3, :, :], m.U_mat, 1, 1)
            χciωk[iω, ik1, ik2, ik3, :, :] .= denom \ @view(g.χ0iωk[iω, ik1, ik2, ik3, :, :])
        end
        if m.prmt.SC_type[end] == 's' # singlet
            for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.bnω
                Viωk[iω, ik1, ik2, ik3, :, :] .= @views(
                    1.5 .* (m.U_mat * g.χiωk[iω, ik1, ik2, ik3, :, :] * m.U_mat) .-
                    0.5 .* (m.U_mat * χciωk[iω, ik1, ik2, ik3, :, :] * m.U_mat)
                )
            end
            e.V_DC .= m.U_mat
        elseif m.prmt.SC_type[end] == 't' # triplet
            for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.bnω
                Viωk[iω, ik1, ik2, ik3, :, :] .= @views(
                    -0.5 .* (m.U_mat * g.χiωk[iω, ik1, ik2, ik3, :, :] * m.U_mat) .-
                    0.5 .* (m.U_mat * χciωk[iω, ik1, ik2, ik3, :, :] * m.U_mat)
                )
            end
            e.V_DC .= 0.0
        end
    end

    # Fourier transform
    Viωr::Array{ComplexF64, 6} = k_to_r(m, Viωk)
    e.Viτr .= ωn_to_τ(m, Bosonic(), Viωr)
end

### Set inital gap Δ0(iω_n, k) --------------------------------------
"""
Set initial guess for gap function according to system symmetry.
The setup is carried out in real space and then FT.
"""
function set_Δiωk!(m::Mesh, g::Gfunction, e::Eliashberg)
    p::Parameters = m.prmt

    ### set basis functions
    Δdx2y2 = Array{ComplexF64, 3}(undef, p.nk1, p.nk2, p.nk3)
    Δdxy = Array{ComplexF64, 3}(undef, p.nk1, p.nk2, p.nk3)
    Δpx = Array{ComplexF64, 3}(undef, p.nk1, p.nk2, p.nk3)
    Δpy = Array{ComplexF64, 3}(undef, p.nk1, p.nk2, p.nk3)
    for ik2 in 1:p.nk2, ik1 in 1:p.nk1
        k1::Float64 = (2π*(ik1-e.iQ[1]/2-1)) / p.nk1
        k2::Float64 = (2π*(ik2-e.iQ[2]/2-1)) / p.nk2
        Δdx2y2[ik1, ik2, :] .= cos(k1) - cos(k2)
        Δdxy[ik1, ik2, :] .= sin(k1) * sin(k2)
        Δpx[ik1, ik2, :] .= sin(k1)
        Δpy[ik1, ik2, :] .= sin(k2)
    end

    ### Set inital gap function according to symmetry
    δ::Float64 = 0.1
    if m.prmt.nspin == 2
        if p.SC_type == "A1"
            e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ (
                Δdx2y2 ⊗ kron(diagm([1, 1, 1, 1]), 1im.*σ2) # singlet: dx2y2
                .+ δ .* Δpx ⊗ kron(diagm([1, -1, -1, 1]), σ1) # triplet: px sz
                .+ δ .* Δpy ⊗ kron(diagm([1, -1, 1, -1]), σ1) # triplet: py sz
            )
        elseif p.SC_type == "A2"
            e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ (
                Δdxy ⊗ kron(diagm([1, 1, 1, 1]), 1im.*σ2) # singlet: dx2y2
                .+ δ .* Δpx ⊗ kron(diagm([1, -1, 1, -1]), σ1) # triplet: px sz
                .+ δ .* Δpy ⊗ kron(diagm([1, -1, -1, 1]), σ1) # triplet: py sz
            )
        elseif p.SC_type == "B1"
            # triplet: py sz
            e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ Δpy ⊗ kron(diagm([1, 1, 1, 1]), σ1)
        elseif p.SC_type == "B2"
            # triplet: px sz
            e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ Δpx ⊗ kron(diagm([1, 1, 1, 1]), σ1)
        end

    elseif m.prmt.nspin == 1
        if p.SC_type == "A1s"
            # singlet: dx2-y2
            e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ Δdx2y2 ⊗ diagm([1, 1, 1, 1])
        elseif p.SC_type == "A2s"
            # singlet: dxy
            e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ Δdxy ⊗ diagm([1, 1, 1, 1])
        elseif p.SC_type == "B1t"
            # triplet: px
            e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ Δpx ⊗ diagm([1, 1, 1, 1])
        elseif p.SC_type == "B2t"
            # triplet: py
            e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ Δpy ⊗ diagm([1, 1, 1, 1])
        end
    end
    e.Δiωk .*= sqrt(m.fnω * p.nk * p.nwan^2) / norm(e.Δiωk)

    e.Δiωk
end

### Set anomalous Green function F(τ, r) --------------------------------------
function set_fiτr!(m::Mesh, g::Gfunction, e::Eliashberg)
    # G(-k, -iωn)
    giωk_invk::Array{ComplexF64, 6} = reverse(
        circshift(g.giωk, (0, -1, -1, -1, 0, 0)),
        dims=(1, 2, 3, 4)
    )
    # G(-k+Q, -iωn)^T
    giωk_invk = permutedims(
        circshift(
            giωk_invk,
            (0, e.iQ[1], e.iQ[2], e.iQ[3], 0, 0)
        ),
        (1, 2, 3, 4, 6, 5)
    )

    fiωk = Array{ComplexF64, 6}(undef, m.fnω, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3, m.prmt.nwan, m.prmt.nwan)
    for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.fnω
        fiωk[iω, ik1, ik2, ik3, :, :] .= @views(
            -g.giωk[iω, ik1, ik2, ik3, :, :] * e.Δiωk[iω, ik1, ik2, ik3, :, :] * giωk_invk[iω, ik1, ik2, ik3, :, :]
        )
    end

    # Fourier transform
    fiωr::Array{ComplexF64, 6} = k_to_r(m, fiωk)
    e.fiτr .= ωn_to_τ(m, Fermionic(), fiωr)

    f_l::Array{ComplexF64, 3} = fit(m.IR_basis_set.smpl_tau_f, e.fiτr[:, 1, 1, 1, :, :], dim=1)
    for ζ2 in 1:m.prmt.nwan, ζ1 in 1:m.prmt.nwan
        e.fiτr_0[ζ1, ζ2] = dot(m.IR_basis_set.basis_f.u(0), @view(f_l[:, ζ1, ζ2]))
    end
end


##### Symmetry arguments (not used for FFLO state) #####
function PointGroup(m::Mesh)
    grp = "C2v"
    order = 4
    irrep::String = m.prmt.SC_type

    character = Vector{Float64}(undef, order)
    if irrep[1:2] == "A1"
        character .= [1.0, 1.0, 1.0, 1.0]
    elseif irrep[1:2] == "A2"
        character .= [1.0, 1.0, -1.0, -1.0]
    elseif irrep[1:2] == "B1"
        character .= [1.0, -1.0, 1.0, -1.0]
    elseif irrep[1:2] == "B2"
        character .= [1.0, -1.0, -1.0, 1.0]
    end

    PointGroup(grp, order, irrep, character)
end

function symmetry_operator(m::Mesh, pg::PointGroup, ik1::Int64, ik2::Int64, ik3::Int64)
    k1::Float64 = (2π*(ik1-1)) / m.prmt.nk1
    k2::Float64 = (2π*(ik2-1)) / m.prmt.nk2

    repres::Vector{Matrix{ComplexF64}} = [
        Matrix{ComplexF64}(undef, m.prmt.nwan, m.prmt.nwan)
        for _ in 1:pg.order
    ]
    id::Vector{Vector{Int64}} = [
        Vector{Int64}(undef, 3)
        for _ in 1:pg.order
    ]

    if m.prmt.nspin == 2
        repres .= [
            Matrix{ComplexF64}(I, m.prmt.nwan, m.prmt.nwan), # E
            kron(σ0, σ1, -1im.*σ3), # C2z
            kron(ComplexF64[0 0 0 cis(-k1); 0 0 cis(k2) 0; 0 1 0 0; cis(-k1+k2) 0 0 0], -1im.*σ1), # Mx
            kron(ComplexF64[0 0 1 0; 0 0 0 cis(k1+k2); cis(k1) 0 0 0; 0 cis(k2) 0 0], -1im.*σ2) # My
        ]
    elseif m.prmt.nspin == 1
        repres .= [
            Matrix{ComplexF64}(I, m.prmt.nwan, m.prmt.nwan), # E
            kron(σ0, σ1), # C2z
            ComplexF64[0 0 0 cis(-k1); 0 0 cis(k2) 0; 0 1 0 0; cis(-k1+k2) 0 0 0], # Mx
            ComplexF64[0 0 1 0; 0 0 0 cis(k1+k2); cis(k1) 0 0 0; 0 cis(k2) 0 0] # My
        ]
    end
    id .= [
        [ik1, ik2, ik3], # E
        [mod(2-ik1, 1:m.prmt.nk1), mod(2-ik2, 1:m.prmt.nk2), ik3], # C2z
        [mod(2-ik1, 1:m.prmt.nk1), ik2, ik3], # Mx
        [ik1, mod(2-ik2, 1:m.prmt.nk2), ik3] # My
    ]

    repres, id
end

"""
Point group projection operator for order parameter
"""
function projection_operator(m::Mesh, pg::PointGroup, yiωk::Array{ComplexF64})
    yiωk_new::Array{ComplexF64} = zeros(ComplexF64, size(yiωk)...)
    for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1
        repres_p, id = symmetry_operator(m, pg, ik1, ik2, ik3)
        repres_m, _  = symmetry_operator(m, pg, mod(2-ik1, 1:m.prmt.nk1), mod(2-ik2, 1:m.prmt.nk2), mod(2-ik3, 1:m.prmt.nk3))
        for iω in 1:m.fnω, ip in 1:pg.order
            jk1::Int64, jk2::Int64, jk3::Int64 = id[ip]
            yiωk_new[iω, ik1, ik2, ik3, :, :] .+= pg.character[ip] .* @views(
                repres_p[ip] * yiωk[iω, jk1, jk2, jk3, :, :] * transpose(repres_m[ip])
            )
        end
    end
    yiωk_new .*= pg.character[1] ./ pg.order

    yiωk_new
end
