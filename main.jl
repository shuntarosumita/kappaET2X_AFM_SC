include("parameters.jl")
include("mesh.jl")
include("gfunction.jl")
include("eliashberg.jl")
include("kpath_extract.jl")
using Dates

function main()
    length(ARGS) < 14 && error("usage: julia main.jl system SC_type h h_load n U U_load T round_it mode nk1 nk2 nk3 wmax")
    system::String = ARGS[1]
    SC_type::String = ARGS[2]
    h::Float64 = parse(Float64, ARGS[3])
    h_load::Float64 = parse(Float64, ARGS[4])
    n_fill::Float64 = parse(Float64, ARGS[5])
    U::Float64 = parse(Float64, ARGS[6])
    U_load::Float64 = parse(Float64, ARGS[7])
    T::Float64 = parse(Float64, ARGS[8])
    round_it::Int64 = parse(Int64, ARGS[9])
    mode::String = ARGS[10]
    nk1::Int64 = parse(Int64, ARGS[11])
    nk2::Int64 = parse(Int64, ARGS[12])
    nk3::Int64 = parse(Int64, ARGS[13])
    wmax::Float64 = parse(Float64, ARGS[14])
    println("nthreads = ", Threads.nthreads())

    ### Initiate parameters -------------------------------------------------------
    start = now()
    p = Parameters(
        system, SC_type, h, n_fill, U, T, round_it, mode; h_load=h_load, U_load=U_load,
        nk1, nk2, nk3, wmax
    )
    save_Parameters(p)
    open(p.Logstr, "a") do log
        println(log, "##################################################")
        println(
            log,
            @sprintf(
                "Parameter set: h = %.3f, n = %.3f, U = %.3f, T = %.4f",
                p.h, p.n_fill, p.U, p.T
            )
        )
        println(
            log,
            "Elapsed time - parameter init: ",
            (now() - start).value / 1000, "s"
        )
    end

    ### Load mesh -----------------------------------------------------------------
    t_mset = now()
    m = Mesh(p)
    open(p.Logstr, "a") do log
        println(log, "fnω, fnτ, iω0_f: $(m.fnω), $(m.fnτ), $(m.iω0_f)")
        println(log, "bnω, bnτ, iω0_b: $(m.bnω), $(m.bnτ), $(m.iω0_b)")
        println(
            log,
            "Elapsed time - Mesh set (tot | module): ",
            (now() - start).value / 1000, "s | ", (now() - t_mset).value / 1000, "s"
        )
    end

    ### The case of skipping Green function calculation
    if minimum(abs.(m.ek)) > 1e-2
        open(p.Logstr, "a") do log
            println(log, "Insulating state => Not calculating Green function.")
            println(log, "##################################################")
        end
        return nothing
    end

    ### Calculate full Green function ---------------------------------------------
    t_gset = now()
    g = Gfunction(m) # initialize Gfunction
    if p.mode == "FLEX"
        FLEXcheck = solve_FLEX!(m, g)
    end
    save_Gfunctions(m, g)
    open(p.Logstr, "a") do log
        println(
            log,
            "Elapsed time - Gfunction calc (tot | module): ",
            (now() - start).value / 1000, "s | ", (now() - t_gset).value / 1000, "s"
        )
    end

    ### Extract quantities along HS path in BZ ------------------------------------
    kpath_extract(m, g)

    ### The case of skipping Eliashberg calculation
    if p.SC_type == "skip"
        open(p.Logstr, "a") do log
            println(log, "SC_type = skip => Not calculating eliashberg equation.")
            println(log, "##################################################")
        end
        return nothing
    elseif p.mode == "FLEX"
        # U, Σ convergence
        if FLEXcheck === missing
            open(p.Logstr, "a") do log
                println(log, "Not calculating eliashberg equation.")
                println(log, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                println(log, "##################################################")
            end
            open(p.Logerrstr, "a") do logerr
                println(logerr, p.err_str_begin * "=> eliashberg skipped.")
            end
            return missing
        end
    elseif p.mode == "RPA"
        if max_eigval_χU(m, g) >= 1.0
            open(p.Logstr, "a") do log
                println(log, "max(χU) > 1 => Not calculating eliashberg equation.")
                println(log, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                println(log, "##################################################")
            end
            open(p.Logerrstr, "a") do logerr
                println(logerr, p.err_str_begin * "=> eliashberg skipped.")
            end
            return missing
        end
    end

    ### Calculate SC parameter --------------------------------------------------
    open(p.Logstr, "a") do log
        println(log, "Move to SC calculation.")
    end
    Qrange = hcat(
        [ik ÷ (nk2÷8+1) for ik in 0:((nk1÷8+1)*(nk2÷8+1)-1)],
        [ik % (nk2÷8+1) for ik in 0:((nk1÷8+1)*(nk2÷8+1)-1)]
    )

    e = Eliashberg(m, g, [0, 0, 0])
    for (iQ1, iQ2) in eachrow(Qrange)
        iQ::Vector{Int64} = [iQ1, iQ2, 0]
        open(p.Logstr, "a") do log
            println(log, "Eliashberg calculation for iQ = $iQ")
        end

        t_eset = now()
        e.iQ .= iQ
        solve_Eliashberg!(m, g, e)
        save_Eliashberg(m, g, e)
        open(p.Logstr, "a") do log
            println(
                log,
                @sprintf(
                    "Done: h = %.3f | n = %.3f | U = %.3f | T = %.4f (%.3f K) | λ = %.6f",
                    p.h, p.n_fill, p.U, p.T, p.T*1.16045*10^4, e.λ
                )
            )
            println(
                log,
                "Elapsed time - Eliashberg calc (tot | module): ",
                (now() - start).value / 1000, "s | ", (now() - t_eset).value / 1000, "s"
            )
        end
        if iQ2 == nk2÷8
            open(p.SC_EV_path, "a") do file
                println(file, "")
            end
        end

        ### Extract quantities along HS path in BZ ------------------------------------
        if e.iQ == [0, 0, 0]
            kpath_extract(m, e)
        end
    end

    open(p.Logstr, "a") do log
        println(log, "##################################################")
    end

    return nothing
end

main()
