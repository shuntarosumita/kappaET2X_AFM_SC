using LinearAlgebra
using Roots
using FFTW
using SparseIR
import SparseIR: Statistics

##############
# Set Mesh for calculation
##############
mutable struct Mesh
    prmt::Parameters

    hk::Array{ComplexF64, 5}
    ek0::Array{Float64, 4}
    ek::Array{Float64, 4}
    uk::Array{ComplexF64, 5}
    μ::Float64
    emin::Float64
    emax::Float64
    W::Float64
    U_mat::Matrix{Float64}

    iω0_f::Int64
    iω0_b::Int64
    fnω::Int64
    fnτ::Int64
    bnω::Int64
    bnτ::Int64
    IR_basis_set::FiniteTempBasisSet
end

function Mesh(p::Parameters)::Mesh
    # Compute Hamiltonian
    hk::Array{ComplexF64, 5} = set_hamiltonian_matrix(p)
    ek0 = Array{Float64, 4}(undef, p.nk1, p.nk2, p.nk3, p.nwan)
    uk = zeros(ComplexF64, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        for is in 1:p.nspin
            range = is:p.nspin:p.nwan
            ek0[ik1, ik2, ik3, range], uk[ik1, ik2, ik3, range, range] = eigen(
                @view(hk[ik1, ik2, ik3, range, range])
            )
        end
        for iband in 1:p.nwan
            id::Int64 = findmax(abs, @view(uk[ik1, ik2, ik3, :, iband]))[2]
            @view(uk[ik1, ik2, ik3, :, iband]) ./= cis(angle(uk[ik1, ik2, ik3, id, iband]))
        end
    end

    # U(1) phase fixing if both TRS and IS are preserved
    if p.h == 0.0
        uk_invk::Array{ComplexF64, 5} = reverse(
            circshift(uk, (-1, -1, -1, 0, 0)),
            dims=(1, 2, 3)
        )

        invid = [3, 4, 1, 2, 7, 8, 5, 6]
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:(p.nk1÷2+1)
            # IS: u_{l s, b σ}(k) = u_{Il s, b σ}(-k)
            for ζ in 1:p.nwan
                @views uk[ik1, ik2, ik3, ζ, :] .= uk_invk[ik1, ik2, ik3, invid[ζ], :]
            end

            # TRS: u_{l s, b σ}(k) = sσ u_{l sbar, b σbar}(-k)^*
            for b in 1:p.norb, σ in 1:p.nspin, α in 1:p.norb, s in 1:p.nspin
                uk[ik1, ik2, ik3, p.nspin*(α-1) + s, p.nspin*(b-1) + σ] = (-1)^(s+σ) * conj(
                    uk_invk[ik1, ik2, ik3, p.nspin*(α-1) + (3-s), p.nspin*(b-1) + (3-σ)]
                )
            end
        end
    end

    μ::Float64 = set_μ(p, ek0)
    ek::Array{Float64, 4} = ek0 .- μ
    emin::Float64 = minimum(ek0)
    emax::Float64 = maximum(ek0)
    W::Float64 = emax - emin

    U_mat::Matrix{Float64} = set_interaction(p)

    # IR basis
    IR_basis_set = FiniteTempBasisSet(p.β, p.wmax, p.IR_tol)

    # lowest Matsubara frequency index
    iω0_f::Int64 = findall(x -> x == FermionicFreq(1), IR_basis_set.smpl_wn_f.sampling_points)[1]
    iω0_b::Int64 = findall(x -> x == BosonicFreq(0), IR_basis_set.smpl_wn_b.sampling_points)[1]

    # the number of sampling point for fermion and boson
    fnω::Int64 = length(IR_basis_set.smpl_wn_f.sampling_points)
    fnτ::Int64 = length(IR_basis_set.smpl_tau_f.sampling_points)
    bnω::Int64 = length(IR_basis_set.smpl_wn_b.sampling_points)
    bnτ::Int64 = length(IR_basis_set.smpl_tau_b.sampling_points)

    return Mesh(
        p, hk, ek0, ek, uk, μ, emin, emax, W, U_mat,
        iω0_f, iω0_b, fnω, fnτ, bnω, bnτ, IR_basis_set
    )
end

"Here the systems Mesh matrix (in orbital basis) has to be set up."
function set_hamiltonian_matrix(p::Parameters)
    ### Model parameters
    ta::Float64, tb::Float64, tp::Float64, tq::Float64 = 0.0, 0.0, 0.0, 0.0
    if p.system == "Cl"
        ta = -0.207
        tb = -0.067
        tp = -0.102
        tq = 0.043
    elseif p.system == "Br"
        ta = 0.196
        tb = 0.065
        tp = 0.105
        tq = -0.039
    elseif p.system == "test"
        ta = -1.0
        tb = -0.5
        tp = -0.45
        tq = 0.15
    end

    ### H_kin without chemical potential
    hk = Array{ComplexF64, 5}(undef, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    h_afm::Matrix{ComplexF64} = kron(diagm([p.h, p.h, -p.h, -p.h]), ComplexF64[1 0; 0 -1])
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        k1::Float64 = (2π*(ik1-1)) / p.nk1
        k2::Float64 = (2π*(ik2-1)) / p.nk2

        ### hopping terms
        h_hop::Matrix{ComplexF64} = [
            (0.0) (ta + tb * cis(-k1)) (tp * (1 + cis(-k1))) (tq * (1 + cis(-k2)))
            (ta + tb * cis(k1)) (0.0)  (tq * (1 + cis(k2))) (tp * (1 + cis(k1)))
            (tp * (1 + cis(k1))) (tq * (1 + cis(-k2))) (0.0) ((ta * cis(k1) + tb) * cis(-k2))
            (tq * (1 + cis(k2))) (tp * (1 + cis(-k1))) ((ta * cis(-k1) + tb) * cis(k2)) (0.0)
        ]

        if p.nspin == 2
            @view(hk[ik1, ik2, ik3, :, :]) .= kron(h_hop, Matrix{ComplexF64}(I, 2, 2)) .+ h_afm
        elseif p.nspin == 1
            @view(hk[ik1, ik2, ik3, :, :]) .= h_hop
        end
    end

    hk
end

"Determining inital mu from bisection method of sum(fermi_k, n) (so for all orbitals!)"
function set_μ(p::Parameters, ek::Array{Float64, 4})
    ### Set electron number for bisection difference
    # n_0 is per orbital
    n_0::Float64 = p.n_fill

    find_zero(
        μ -> 2/p.nspin * calc_electron_density(p, ek, μ) - n_0,
        (3*minimum(ek), 3*maximum(ek)), Bisection()
    )
end

function calc_electron_density(p::Parameters, ek::Array{Float64, 4}, μ::Float64)
    E = fill(one(Float64), size(ek)...)
    sum(E ./ (E .+ exp.(p.β .* (ek .- μ)))) / (p.nk * p.norb)
end

function set_interaction(p::Parameters)
    U_mat::Matrix{Float64} = zeros(Float64, p.nwan^2, p.nwan^2)
    if p.nspin == 2
        for α in 1:p.norb, s in 1:p.nspin
            sbar::Int = 3 - s

            # udud, dudu
            ζ12::Int = p.nwan*((p.nspin*(α-1) + s)-1) + p.nspin*(α-1) + sbar
            ζ34::Int = p.nwan*((p.nspin*(α-1) + s)-1) + p.nspin*(α-1) + sbar
            U_mat[ζ12, ζ34] += p.U

            # uudd, dduu
            ζ12 = p.nwan*((p.nspin*(α-1) + s)-1) + p.nspin*(α-1) + s
            ζ34 = p.nwan*((p.nspin*(α-1) + sbar)-1) + p.nspin*(α-1) + sbar
            U_mat[ζ12, ζ34] -= p.U
        end

    elseif p.nspin == 1
        for α in 1:p.norb
            # udud, dudu
            ζ12::Int = p.nwan*(α-1) + α
            ζ34::Int = p.nwan*(α-1) + α
            U_mat[ζ12, ζ34] += p.U
        end
    end

    U_mat
end


"Return sampling object for given statistic"
function smpl_obj(mesh::Mesh, statistics::Statistics)
    if statistics == Fermionic()
        smpl_τ = mesh.IR_basis_set.smpl_tau_f
        smpl_ωn  = mesh.IR_basis_set.smpl_wn_f
    elseif statistics == Bosonic()
        smpl_τ = mesh.IR_basis_set.smpl_tau_b
        smpl_ωn  = mesh.IR_basis_set.smpl_wn_b
    end    
    return smpl_τ, smpl_ωn
end

# Fourier transformation
"Fourier transform from τ to iω_n via IR basis"
function τ_to_ωn(mesh::Mesh, statistics::Statistics, obj_τ::Array{ComplexF64, n}) where n
    smpl_τ, smpl_ωn = smpl_obj(mesh, statistics)

    obj_l = fit(smpl_τ, obj_τ, dim=1)
    obj_ωn::Array{ComplexF64, n} = evaluate(smpl_ωn, obj_l, dim=1)
    return obj_ωn
end

"Fourier transform from iω_n to τ via IR basis"
function ωn_to_τ(mesh::Mesh, statistics::Statistics, obj_ωn::Array{ComplexF64, n}) where n
    smpl_τ, smpl_ωn = smpl_obj(mesh, statistics)

    obj_l = fit(smpl_ωn, obj_ωn, dim=1)
    obj_τ::Array{ComplexF64, n} = evaluate(smpl_τ, obj_l, dim=1)
    return obj_τ
end

"Fourier transform from k-space to real space"
function k_to_r(mesh::Mesh, obj_k::Array{ComplexF64, n}) where n
    obj_r::Array{ComplexF64, n} = fft(obj_k, [2, 3, 4])
    return obj_r
end

"Fourier transform from real space to k-space"
function r_to_k(mesh::Mesh, obj_r::Array{ComplexF64, n}) where n
    obj_k::Array{ComplexF64, n} = ifft(obj_r, [2, 3, 4]) / mesh.prmt.nk
    return obj_k
end
