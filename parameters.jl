using Printf
using HDF5

"""
Setting parameters for current calculation
"""
mutable struct Parameters
    mode::String
    mix::Float64
    round_it::Int64

    SC_type::String

    IR_tol::Float64
    g_sfc_tol::Float64
    SC_sfc_tol::Float64

    system::String
    nk1::Int64
    nk2::Int64
    nk3::Int64
    nk::Int64
    T::Float64
    β::Float64
    wmax::Float64
    n_fill::Float64
    nspin::Int64
    norb::Int64
    nwan::Int64
    U::Float64
    h::Float64
    pos::Vector{Vector{Float64}}

    Logstr::String
    Logerrstr::String
    err_str_begin::String
    savepath::String
    loadpath::String
    plot_savepath::String
    plot_loadpath::String
    
    BSE_EV_path::String
    SC_EV_path::String
    SC_EV_path_neg::String
end

function Parameters(
        system::String,
        SC_type::String,
        h::Float64,
        n::Float64,
        U::Float64,
        T::Float64,
        round_it::Int64,
        mode::String;
        h_load::Float64=0.0,
        U_load::Float64=1.0,
        nk1::Int64,
        nk2::Int64,
        nk3::Int64,
        wmax::Float64
    )::Parameters

    # General settings
    mix::Float64 = 0.2 # Value of how much of the new G is to be used

    # Cutoffs/accuracy
    IR_tol::Float64 = 1e-15
    g_sfc_tol::Float64 = 1e-6
    SC_sfc_tol::Float64 = 1e-5

    # Physical quantities
    nk::Int64 = nk1 * nk2 * nk3
    β::Float64 = 1/T
    n_fill::Float64 = n
    nspin::Int64 = 2
    norb::Int64 = 4
    nwan::Int64 = nspin * norb

    # coordinate of the molecules inside the unit cell
    ## choosing Shastry-Sutherland lattice
    pos::Vector{Vector{Float64}} = [
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0], [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.0], [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0], [0.0, 0.5, 0.0]
    ]

    ### Log options
    Log_name::String = Printf.format(
        Printf.Format("log_k-$(system)/Log_h_%.3f_n_%.3f_U_%.3f_T_%.4f"),
        h, n_fill, U, T
    )
    Logstr::String = Log_name * ".txt"
    Logerrstr::String = Log_name * "_err.txt"
    err_str_begin::String = @sprintf(
        "System h = %.3f | n = %.3f | U = %.3f | T = %.4f : ",
        h, n_fill, U, T
    )

    ### Setting saving options
    calc_name::String = "Odata_k-$(system)/data_h_%.3f_n_%.3f_U_%.3f_T_%.4f.h5"
    plot_name::String = "Odata_k-$(system)/plot_data_h_%.3f_n_%.3f_U_%.3f_T_%.4f.h5"

    # formatting middle string
    savepath::String = Printf.format(Printf.Format(calc_name), h, n_fill, U, T)
    loadpath::String = Printf.format(Printf.Format(calc_name), h_load, n_fill, U_load, T)
    plot_savepath::String = Printf.format(Printf.Format(plot_name), h, n_fill, U, T)
    plot_loadpath::String = Printf.format(Printf.Format(plot_name), h_load, n_fill, U_load, T)

    # eigenvalue strings
    BSE_EV_path::String = Printf.format(
        Printf.Format("BSE_kernel_EV_k-$(system)/max_ev_n_%.3f_U_%.3f.txt"),
        n_fill, U
    )
    SC_EV_path::String = Printf.format(
        Printf.Format("SC_EV_k-$(system)/$(SC_type)_lam_h_%.3f_n_%.3f_U_%.3f.txt"),
        h, n_fill, U
    )
    SC_EV_path_neg::String = Printf.format(
        Printf.Format("SC_EV_k-$(system)/$(SC_type)_lam_h_%.3f_n_%.3f_U_%.3f_negative.txt"),
        h, n_fill, U
    )

    return Parameters(
        mode, mix, round_it,
        SC_type,
        IR_tol, g_sfc_tol, SC_sfc_tol,
        system, nk1, nk2, nk3, nk, T, β, wmax, n_fill, nspin, norb, nwan, U, h, pos,
        Logstr, Logerrstr, err_str_begin, savepath, loadpath, plot_savepath, plot_loadpath,
        BSE_EV_path, SC_EV_path, SC_EV_path_neg
    )
end

function save_Parameters(p::Parameters)
    sp_dir::String = "Odata_k-$(p.system)"

    ### Generate directories/h5 file if not exist
    isdir("log_k-$(p.system)") || mkdir("log_k-$(p.system)")
    isdir("SC_EV_k-$(p.system)") || mkdir("SC_EV_k-$(p.system)")
    isdir("BSE_kernel_EV_k-$(p.system)") || mkdir("BSE_kernel_EV_k-$(p.system)")
    isdir(sp_dir) || mkdir(sp_dir)

    isfile(p.savepath) || (
        h5open(p.savepath, "w") do file
            write(file, "SystemName", "κ-$(p.system)")
            write(file, "N_k1", p.nk1)
            write(file, "N_k2", p.nk2)
            write(file, "N_k3", p.nk3)
            write(file, "IR_wmax", p.wmax)
            write(file, "IR_tol", p.IR_tol)
            write(file, "g_sfc_tol", p.g_sfc_tol)
            write(file, "SC_sfc_tol", p.SC_sfc_tol)
            write(file, "n_fill", p.n_fill)
            write(file, "T", p.T)
            write(file, "U", p.U)
            write(file, "h", p.h)
        end
    )
end
