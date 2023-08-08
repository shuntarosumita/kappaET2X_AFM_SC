using LinearAlgebra
using Roots
using HDF5
using FLoops
import SparseIR: valueim

##########################################################################
##### Main code: Calculate Greens function within FLEX approximation #####
##########################################################################
mutable struct Gfunction
    U_pval::Float64
    μ::Float64

    giωk::Array{ComplexF64, 6}
    giτr::Array{ComplexF64, 6}
    giτr_0::Array{ComplexF64, 2}
    χ0iωk::Array{ComplexF64, 6}
    χiωk::Array{ComplexF64, 6}
    Σiωk::Array{ComplexF64, 6}
end

my_iter(iters) = Tuple.(CartesianIndices(iters))

"Initialize Green function"
function Gfunction(m::Mesh)::Gfunction
    p::Parameters = m.prmt

    ##### Set initial round (bare quantities)
    ## Initial guess for μ, Σ
    μ::Float64 = m.μ
    Σiωk::Array{ComplexF64, 6} = zeros(ComplexF64, m.fnω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    if p.round_it == 0
        open(p.Logstr, "a") do log
            println(log, "G convergence from ZERO.")
        end
    else
        open(p.Logstr, "a") do log
            println(log, "G convergence from pre converged G.")
        end
        h5open(p.loadpath, "r") do file
            μ = read(file, "gfunction/μ")
            Σiωk .= read(file, "gfunction/Σiωk")
        end
    end

    giωk = Array{ComplexF64, 6}(undef, m.fnω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    giτr = Array{ComplexF64, 6}(undef, m.fnτ, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    giτr_0 = Array{ComplexF64, 2}(undef, p.nwan, p.nwan)
    χ0iωk = Array{ComplexF64, 6}(undef, m.bnω, p.nk1, p.nk2, p.nk3, p.nwan^2, p.nwan^2)
    χiωk = Array{ComplexF64, 6}(undef, m.bnω, p.nk1, p.nk2, p.nk3, p.nwan^2, p.nwan^2)

    g = Gfunction(p.U, μ, giωk, giτr, giτr_0, χ0iωk, χiωk, Σiωk)

    ## Set G(iωn, k), χ_0(iωn, k), χ(iωn, k)
    if p.mode == "RPA" && p.round_it != 0
        h5open(p.loadpath, "r") do file
            g.giωk .= read(file, "gfunction/giωk")
            g.χ0iωk .= read(file, "gfunction/χ0iωk")
        end
    else
        set_giωk!(m, g, μ)
        set_giτr!(m, g)
        set_χ0iωk!(m, g)
    end
    set_χiωk!(m, g)

    return g
end

function save_Gfunctions(m::Mesh, g::Gfunction)
    ### Calculate maximal magnetic/charge eigenvalue
    open(m.prmt.Logstr, "a") do log
        println(log, "Extract largest BSE kernel values.")
    end
    BSE_max::Float64 = max_eigval_χU(m, g)

    open(m.prmt.BSE_EV_path, "a") do file
        println(
            file,
            "$(m.prmt.T) $(m.prmt.h) $BSE_max"
        )
    end

    open(m.prmt.Logstr, "a") do log
        println(
            log,
            "### Maximal BSE kernel value χU = $BSE_max"
        )
    end

    ##### Save results
    open(m.prmt.Logstr, "a") do log
        println(log, "Saving all data now...")
    end

    h5open(m.prmt.savepath, "cw") do file
        haskey(file, "gfunction") && delete_object(file, "gfunction")
        group = create_group(file, "gfunction")
        group["giωk"] = g.giωk
        group["χ0iωk"] = g.χ0iωk
        group["Σiωk"] = g.Σiωk
        group["μ"] = g.μ
        group["BSE_max"] = BSE_max
    end

    open(m.prmt.Logstr, "a") do log
        println(log, "Done.")
    end
end

####################
# Loop solving instance
####################
"Gfunction.solve_FLEX!() executes FLEX loop until convergence"
function solve_FLEX!(m::Mesh, g::Gfunction)
    p::Parameters = m.prmt

    ##### Set parameters for U convergence
    g.U_pval::Float64 = p.U
    ΔU::Float64 = p.U / 2.0
    U_it::Int64 = 1

    div_check::Float64 = max_eigval_χU(m, g)
    conv_tol::Float64 = p.g_sfc_tol
    sfc_check::Float64 = 1.0

    # perform loop until convergence is reached:
    while div_check >= 1.0 || abs(g.U_pval - p.U) > 1e-10 || U_it == 1
        # Safety check for too long running calculations
        if U_it == 100
            open(p.Logstr, "a") do log
                println(log, "U iteration reached step 100. Everything okay?")
            end
            open(p.Logerrstr, "a") do log
                println(log, p.err_str_begin * "U iteration reached step 100")
            end
        end

        # If it's not good already after one cycle, reset U
        if abs(g.U_pval - p.U) > 1e-10
            p.U += ΔU
            m.U_mat = set_interaction(p)
            div_check = max_eigval_χU(m, g)
        end

        ##### Setting new U if max(χ0*U) >= 1
        div_check, ΔU = U_renormalization!(m, g, div_check, ΔU)

        #### Convergence cycle of FLEX self-energy for given U
        # Setting of convergence tolerance and iteration number
        if abs(g.U_pval - p.U) > 1e-10
            conv_tol = 1e-4
            sfc_it_max = 200
        else
            conv_tol = p.g_sfc_tol
            sfc_it_max = 400
        end

        # Convergence loop
        for it_sfc in 1:sfc_it_max
            sfc_check = loop!(m, g)

            open(p.Logstr, "a") do log
                println(log, "$it_sfc: $sfc_check")
            end

            sfc_check <= conv_tol && break
        end

        U_it += 1
    end

    ##### Security convergence check
    ### U convergence
    if abs(g.U_pval - p.U) > 1e-10
        open(p.Logstr, "a") do log
            println(log, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            println(log, "U is not initial input. Stopping gfunction.")
        end
        open(p.Logerrstr, "a") do logerr
            println(logerr, p.err_str_begin * "U != U_init | gfunction stopped.")
        end

        return missing
    end

    ### giωk convergence
    if sfc_check > conv_tol
        open(p.Logstr, "a") do log
            println(log, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            println(log, "giωk not converged. Stopping gfunction.")
        end
        open(p.Logerrstr, "a") do logerr
            println(
                logerr,
                Printf.format(
                    Printf.Format(p.err_str_begin * "giωk not converged (diff = %.6f)) | gfunction stopped."),
                    sfc_check
                )
            )
        end
        return missing
    end

    open(p.Logstr, "a") do log
        println(log, "Self consistency loop finished!")
    end

    return nothing
end

"FLEX loop"
function loop!(m::Mesh, g::Gfunction)
    giωk_old::Array{ComplexF64, 6} = copy(g.giωk)

    set_giτr!(m, g)
    set_χ0iωk!(m, g)
    set_χiωk!(m, g)

    set_Σiωk!(m, g)

    set_μ_from_giωk!(m, g)
    set_giωk!(m, g, g.μ)
    symmetrize_giωk!(g)
    sfc_check::Float64 = maximum(abs.(g.giωk .- giωk_old))

    # Mixing: Change values if needed!
    g.giωk = m.prmt.mix .* g.giωk .+ (1-m.prmt.mix) .* giωk_old

    sfc_check
end


####################
# U renormalization loop instance
####################
"Loop for renormalizing U if Stoner enhancement max(χ0*U) >= 1."
function U_renormalization!(m::Mesh, g::Gfunction, div_check::Float64, ΔU::Float64)::Tuple{Float64, Float64}
    open(m.prmt.Logstr, "a") do log
        println(
            log, "### Check for renormalization max(χ×U): ", div_check,
            ", U = ", m.prmt.U
        )
    end

    # renormalization loop
    while div_check >= 1.0
        # remormalize U such that max(χ0*U) < 1
        m.prmt.U -= ΔU
        ΔU /= 2.0
        println("New U set to ", m.prmt.U)

        m.U_mat = set_interaction(m.prmt)
        div_check = max_eigval_χU(m, g)
    end

    open(m.prmt.Logstr, "a") do log
        println(log, "New U value: $(m.prmt.U) with max(χ×U) = $div_check")
    end

    div_check, ΔU
end


####################
# Calculate chemical potential via bisection method from G(k, iω_n).
####################
### Set from Greens function ----------------------------------------------
"Determining iteration mu from bisection method of 1 + sum(giωk)"
function set_μ_from_giωk!(m::Mesh, g::Gfunction)
    ### Set electron number for bisection difference
    # n_0 is per electron orbital!
    n_0::Float64 = m.prmt.n_fill
    g.μ = find_zero(
        μ -> 2/m.prmt.nspin * calc_electron_density_from_giωk(m, g, μ) - n_0,
        (m.emax + 2*m.W, m.emin - 2*m.W), Bisection()
    )
end

#--------------------------------------------------------------------------
function calc_electron_density_from_giωk(m::Mesh, g::Gfunction, μ::Float64)
    set_giωk!(m, g, μ)
    giω::Array{ComplexF64, 3} = dropdims(sum(g.giωk, dims=(2, 3, 4)), dims=(2, 3, 4)) ./ m.prmt.nk
    trgiω::Vector{ComplexF64} = [
        sum(giω[iω, ζ, ζ] for ζ in 1:m.prmt.nwan)
        for iω in 1:m.fnω
    ] ./ m.prmt.norb
    g_l::Vector{ComplexF64} = fit(m.IR_basis_set.smpl_wn_f, trgiω, dim=1)
    g_τ0::ComplexF64 = dot(m.IR_basis_set.basis_f.u(0), g_l)

    m.prmt.nspin + real(g_τ0)
end


####################
### Set functions for self consistency loop.
# set_giωk! : m, g | calculates G(iω_n, k)
# set_giτr! : m, g | calculates G(τ, r) via FFT + SparseIR on fermionic τ
# set_χ0iωk! : m, g | calculates χ_0(iν_m, q) via G(τ, r) and FFT + SparseIR
# set_χiωk! : m, g | calculates χ(iν_m, q)
# calc_Viτr : m, g | calculates V_n(τ, r) on fermionic τ via χ_0, FFT + SparseIR
# set_Σiωk! : m, g | calculates Σ(iω_n, k) via V and G
####################
### Set G(iω_n, k) --------------------------------------------------------
"Calculate Green function G(iω, k)"
function set_giωk!(m::Mesh, g::Gfunction, μ::Float64)
    id = Matrix{ComplexF64}(I, m.prmt.nwan, m.prmt.nwan)
    for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.fnω
        iν::ComplexF64 = valueim(m.IR_basis_set.smpl_wn_f.sampling_points[iω], m.prmt.β)
        g.giωk[iω, ik1, ik2, ik3, :, :] .= @views(
            (iν + μ) .* id
            .- m.hk[ik1, ik2, ik3, :, :] .- g.Σiωk[iω, ik1, ik2, ik3, :, :]
        ) \ I
    end

    g.giωk
end

function symmetrize_giωk!(g::Gfunction)
    # G(k, iωn) = G(-k, iωn)^T
    giωk_invk::Array{ComplexF64, 6} = permutedims(
        reverse(
            circshift(g.giωk, (0, -1, -1, -1, 0, 0)),
            dims=(2, 3, 4)
        ),
        (1, 2, 3, 4, 6, 5)
    )
    g.giωk .= (g.giωk .+ giωk_invk) ./ 2.0
end

### Set G(τ, r) ---------------------------------------------------------
"Calculate real space Green function G(τ, r) [for calculating χ0 and Σ]"
function set_giτr!(m::Mesh, g::Gfunction)
    # Fourier transform
    giωr::Array{ComplexF64, 6} = k_to_r(m, g.giωk)
    g.giτr .= ωn_to_τ(m, Fermionic(), giωr)

    g_l::Array{ComplexF64, 3} = fit(m.IR_basis_set.smpl_tau_f, g.giτr[:, 1, 1, 1, :, :], dim=1)
    for ζ2 in 1:m.prmt.nwan, ζ1 in 1:m.prmt.nwan
        g.giτr_0[ζ1, ζ2] = dot(m.IR_basis_set.basis_f.u(0), @view(g_l[:, ζ1, ζ2]))
    end
end

### Set χ_0(iν_m, q) ----------------------------------------------------
"Calculate irreducible susciptibility χ0(iν, q)"
function set_χ0iωk!(m::Mesh, g::Gfunction)
    # -G(-τ, -r) = G(β-τ, -r)
    giτr_invr::Array{ComplexF64, 6} = reverse(
        circshift(g.giτr, (0, -1, -1, -1, 0, 0)),
        dims=(1, 2, 3, 4)
    )

    χ0iτr = zeros(ComplexF64, m.bnτ, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3, m.prmt.nwan^2, m.prmt.nwan^2)
    for ζ4 in 1:m.prmt.nwan, ζ3 in 1:m.prmt.nwan, ζ2 in 1:m.prmt.nwan, ζ1 in 1:m.prmt.nwan
        ζ12::Int64 = m.prmt.nwan * (ζ1-1) + ζ2
        ζ34::Int64 = m.prmt.nwan * (ζ3-1) + ζ4
        χ0iτr[:, :, :, :, ζ12, ζ34] .+= @views(
            g.giτr[:, :, :, :, ζ1, ζ3] .* giτr_invr[:, :, :, :, ζ4, ζ2]
        )
    end

    # Fourier transform
    χ0iτk::Array{ComplexF64, 6} = r_to_k(m, χ0iτr)
    g.χ0iωk .= τ_to_ωn(m, Bosonic(), χ0iτk)
end

### Set χ(iν_m, q) ----------------------------------------------------
"Calculate generalized susciptibility χ(iν, q)"
function set_χiωk!(m::Mesh, g::Gfunction)
    # generalized susceptibility
    id = Matrix{ComplexF64}(I, m.prmt.nwan^2, m.prmt.nwan^2)
    for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.bnω
        denom::Matrix{ComplexF64} = copy(id)
        @views mul!(denom, g.χ0iωk[iω, ik1, ik2, ik3, :, :], m.U_mat, -1, 1)
        g.χiωk[iω, ik1, ik2, ik3, :, :] .= denom \ @view(g.χ0iωk[iω, ik1, ik2, ik3, :, :])
    end

    g.χiωk
end

### V_n(τ, r) -------------------------------------------------------------
"Calculate interaction V_n(τ, r) from RPA-like generalized susceptibility for calculating Σiωk"
function calc_Viτr(m::Mesh, g::Gfunction)
    Viωk = Array{ComplexF64, 6}(undef, m.bnω, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3, m.prmt.nwan^2, m.prmt.nwan^2)
    if m.prmt.nspin == 2
        for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.bnω
            Viωk[iω, ik1, ik2, ik3, :, :] .= @views(
                m.U_mat * (g.χiωk[iω, ik1, ik2, ik3, :, :] .- 0.5.*g.χ0iωk[iω, ik1, ik2, ik3, :, :]) * m.U_mat
            )
        end

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
        for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.bnω
            Viωk[iω, ik1, ik2, ik3, :, :] .= @views(
                1.5 .* (m.U_mat * (g.χiωk[iω, ik1, ik2, ik3, :, :] .- 0.5.*g.χ0iωk[iω, ik1, ik2, ik3, :, :]) * m.U_mat) .+
                0.5 .* (m.U_mat * (χciωk[iω, ik1, ik2, ik3, :, :] .- 0.5.*g.χ0iωk[iω, ik1, ik2, ik3, :, :]) * m.U_mat)
            )
        end
    end

    # Constant Hartree Term V ~ U needs to be treated extra, since they cannot be modeled by the IR basis.
    # In the single-band case, the Hartree term can be absorbed into the chemical potential.
    V_DC::Array{ComplexF64, 2} = copy(m.U_mat)

    # Fourier transform
    Viωr::Array{ComplexF64, 6} = k_to_r(m, Viωk)
    Viτr::Array{ComplexF64, 6} = ωn_to_τ(m, Bosonic(), Viωr)

    Viτr, V_DC
end

### Σ(iω_n, k) --------------------------------------------------------
"Calculate self-energy Σ(iω, k)"
function set_Σiωk!(m::Mesh, g::Gfunction)
    # get V
    Viτr::Array{ComplexF64, 6}, V_DC::Array{ComplexF64, 2} = calc_Viτr(m, g)

    Σiτr = zeros(ComplexF64, m.fnτ, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3, m.prmt.nwan, m.prmt.nwan)
    for ζ4 in 1:m.prmt.nwan, ζ3 in 1:m.prmt.nwan, ζ2 in 1:m.prmt.nwan, ζ1 in 1:m.prmt.nwan
        ζ12::Int64 = m.prmt.nwan * (ζ1-1) + ζ2
        ζ34::Int64 = m.prmt.nwan * (ζ3-1) + ζ4
        Σiτr[:, :, :, :, ζ1, ζ3] .+= @views(
            Viτr[:, :, :, :, ζ12, ζ34] .* g.giτr[:, :, :, :, ζ2, ζ4]
        )
    end

    # Fourier transform
    Σiτk::Array{ComplexF64, 6} = r_to_k(m, Σiτr)
    g.Σiωk .= τ_to_ωn(m, Fermionic(), Σiτk)

    # double counting term (absorbed into the chemical potential)
    # for ζ4 in 1:m.prmt.nwan, ζ3 in 1:m.prmt.nwan, ζ2 in 1:m.prmt.nwan, ζ1 in 1:m.prmt.nwan
    #     ζ12::Int64 = m.prmt.nwan * (ζ1-1) + ζ2
    #     ζ34::Int64 = m.prmt.nwan * (ζ3-1) + ζ4
    #     g.Σiωk[:, :, :, :, ζ1, ζ3] .+= V_DC[ζ12, ζ34] * g.giτr_0[ζ2, ζ4] / m.prmt.nk
    # end

    g.Σiωk
end

##############
# Function for calculating max eig(χU) for diverging check
##############
"Calculate max{eig(χ0iωk×U)} as a measure for divergence check in χiωk"
function max_eigval_χU(m::Mesh, g::Gfunction)
    X_eig = Array{ComplexF64}(undef, m.bnω, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3, m.prmt.nwan^2)
    @floop for (ik3, ik2, ik1, iω) in my_iter((1:m.prmt.nk3, 1:m.prmt.nk2, 1:m.prmt.nk1, 1:m.bnω))
        X_eig[iω, ik1, ik2, ik3, :] .= eigvals(@view(g.χ0iωk[iω, ik1, ik2, ik3, :, :]) * m.U_mat)
    end

    id = argmax(real.(X_eig))
    if abs(imag(X_eig[id])) > 1e-10
        open(m.prmt.Logstr, "a") do log
            println(log, "!!!!!! Imaginary part of χ eig_val is very large: $(abs(imag(X_eig[id]))) !!!!!!")
        end
    end

    real(X_eig[id])
end