module SquidData

using CSV;
using DataFrames;
using Glob;
using PyPlot;
using GLM;
using Statistics;
using LsqFit;
using Loess;
using FFTW;
using NumericalIntegration;
using InvertedIndices;

# ---------------- Read pre-processed magnetic moment data -------------------

""" Read a MPMS data file and return a DataFrame. """
function read_data(path::String)
    data = rename((CSV.File(path, header=31) |> DataFrame!)[:, ["Field (Oe)","Temperature (K)", "Long Moment (emu)"]], [:Field, :Temperature, :Moment]);
    data.Field /= 1e4;
    data.Moment /= 1e-6;
    return data
end

# ----------------- Read and process raw SQUID voltage scans -------------------

""" Read a MPMS raw data file and return a DataFrame """
function read_raw(path::String)
    cols = ["Field (Oe)", "Start Temperature (K)", "End Temperature (K)", "Scan", "Rejected", "Position (cm)",
            "Long Voltage", "Long Average Voltage", "Long Detrended Voltage", "Long Demeaned Voltage",
            "Long Reg Fit", "Long Detrended Fit", "Long Demeaned Fit", "Long Scaled Response", "Long Avg. Scaled Response"]
    new_cols = [:Field, :StartTemperature, :EndTemperature, :Scan, :Rejected, :Position,
                :Voltage, :AvgVoltage, :DetrVoltage, :DemVoltage,
                :RegFit, :DetrFit, :DemFit, :ScaledVoltage, :AvgScaledVoltage]
    data = rename((CSV.File(path, header=31) |> DataFrame!)[:, cols], new_cols);
    return data
end

""" Read a MPMS raw data file, fit all raw scans and return a DataFrame with the resulting magnetic moment data. """
function read_and_fit_raw(path::String)
    data = fit_raw(process_raw(SquidData.read_raw(path))[1])
    data.Field /= 1e4
    data.Moment /=1e-6
    return data
end

""" Process SQUID raw data as returned by the read_raw function, such that it can be further processed. """
function process_raw(data::DataFrame)
    data[!, :Temperature] = vec(mean(Array(data[!, [:StartTemperature, :EndTemperature]]), dims=2))
    nscans = maximum(data.Scan)
    npps = 0
    while data.Scan[npps+1] == data.Scan[1]
        npps += 1
    end
    #nmeas = Int(nrow(data) / (nscans*npps))

    data[!, :MeasIdx] = zeros(Int64, nrow(data))
    data[!, :PointIdx] = zeros(Int64, nrow(data))
    curmeas = 1
    curscan = 1
    curpoint = 1
    for i in 1:nrow(data)
        if data[i, :Scan] == curscan + 1
            curscan += 1
            curpoint = 1
        elseif data[i, :Scan] > curscan
            println("Raw data is missing $(data[i, :Scan]-curscan-1) scans!")
            curscan = data[i, :Scan]
            curpoint = 1
        elseif data[i, :Scan] < curscan
            curmeas += 1
            curscan = 1
            curpoint = 1
        end

        data[i, :MeasIdx] = curmeas
        data[i, :PointIdx] = curpoint
        curpoint += 1
    end
    nmeas = curmeas

    fit = select(data, [:MeasIdx, :PointIdx, :AvgVoltage, :DetrVoltage, :DemVoltage, :RegFit, :DetrFit, :DemFit, :AvgScaledVoltage])[data.Scan .== nscans, :]

    raw = select(data, [:MeasIdx, :Scan, :PointIdx, :Position, :Temperature, :Field, :Voltage, :ScaledVoltage])
    raw = transform(groupby(raw, [:MeasIdx, :Scan]),
                    :Voltage => (V->V .- mean(V)) => :DemVoltage,
                    [:Voltage, :PointIdx] => ((V,i)->V - i*GLM.coef(lm(hcat(ones(length(i)),i), collect(V)))[2]) => :DetrVoltage)
    avg = combine(groupby(raw, [:MeasIdx, :PointIdx]), names(raw, Not([:MeasIdx, :PointIdx])) .=> mean .=> names(raw, Not([:MeasIdx, :PointIdx])))
    avg[!, :Position] .-= (maximum(avg[!, :Position]) - minimum(avg[!, :Position]))/2.0
    avg[!, :CalibrationFactor] = avg[!, :ScaledVoltage] ./ avg[!, :Voltage]
    #avg = detrend(demean(avg))
    return avg, fit
end

""" Reframe raw SQUID scans. """
function reframe_scans(avg::DataFrame; max_pos=1.0)
    return avg[abs.(avg[:,:Position]) .<= abs(max_pos), :]
end

""" Filter out bad points from raw SQUID scan data. """
function filter_bad_points(avg::DataFrame; threshold=0.05)
    avg[!, :BadPoint] = zeros(Union{eltype(avg[!, :Voltage]), Missing}, nrow(avg))#convert(Vector{Union{eltype(avg[!, :Voltage]),Missing}}, avg[!, :Voltage])
    nbadpoints = 0
    for gdf in groupby(avg, :MeasIdx)
        # Filter bad points
        for i in 1:nrow(gdf)-5
            if std(gdf[i:i+5, :Voltage]) < 0.05
                bad_pos = gdf[i, :Position]
                println("Found bad point at position", bad_pos)
                gdf[abs.(gdf[:, :Position]) .>= abs(bad_pos), :BadPoint] .= missing
                nbadpoints += 1
                break
            end
        end
    end
    if nbadpoints > 0
        dropmissing!(avg, :BadPoint)
    end
    return nbadpoints, avg
end

""" Detrend the raw SQUID scan data.  """
function detrend(avg::DataFrame)
    # Detrend
    return transform(
        groupby(avg, :MeasIdx),
        [:DemVoltage, :PointIdx] =>
            function(V, i)
                V - i*GLM.coef(lm(hcat(ones(length(i)),i), collect(V)))[2]
            end => :DetrVoltage
    )
end

"""Demean the raw SQUID scan data."""
function demean(avg::DataFrame)
    # Demean
    return transform(
        groupby(avg, :MeasIdx),
        [:Voltage] =>
            function(dV)
                dV .- mean(dV)
            end => :DemVoltage
    )
end

""" Fit the raw SQUID scans using the standard magnetic dipole model. """
function fit_raw(avg::DataFrame)
    LAMBDA = 1.519  # SQUID coil separation (cm)
    R = 0.97        # SQUID coil radius (cm)

    @. model(x, p) = p[1] + p[2] * x + p[3] * (2 * (R^2 + x^2)^(-3/2) - (R^2 + (x+LAMBDA)^2)^(-3/2) - (R^2 + (x-LAMBDA)^2)^(-3/2))
    p0 = [1.0, 0.0, 0.0]
    combine(groupby(avg, :MeasIdx),
            [:DetrVoltage, :CalibrationFactor, :Position] => ((V, c, x) -> coef(curve_fit(model, x, V.*c, p0))[3]) => :Moment,
            :Temperature => mean => :Temperature,
            :Field => mean => :Field
    )
end

""" Fit the raw SQUID scan using the deconvolution algorithm. """
function raw_data_deconvolution(avg::DataFrame; η=0.1, niter=100)
    R = 0.97
    LAMBDA = 1.519

    nmeas = maximum(avg[!, :MeasIdx])

    X = reshape(avg[:, :Position], nmeas, :)
    V = reshape(avg[:, :Voltage], nmeas, :)
    V_ft = fft(V, 2)
    D = V * R^2
    D_ft = fft(D, 2)
    M_ft = V_ft ./ D_ft
    M = real(ifft(M_ft, 2))

    for it in 1:niter
        F = zeros(size(V))
        for i in 1:size(F, 2)
            xi = X .- X[:, i]
            F .+= M[:, i] .* (2 .*(R^2 .+ xi.^2).^(-3/2) .- (R^2 .+ (xi.+LAMBDA).^2).^(-3/2) .- (R^2 .+ (xi.-LAMBDA).^2).^(-3/2))
        end

        E = D .- F
        E_ft = fft(E, 2)
        U_ft = V_ft ./ E_ft
        U = real(ifft(E_ft, 2))

        M .+= η.*U
    end
    m = [integrate(X[i,:], M[i,:]) for i in 1:nmeas]
    avg[!, :MomentDistribution] = reshape(M, :)
    res = combine(groupby(avg, :MeasIdx),
                  :Temperature => mean => :Temperature,
                  :Field => mean => :Field)
    res[!, :Moment] = m
    return res
end

function make_model()
    Chain(Dense(64, 64, x->x.^(-3//2)), Dense(64, 32), Dense(32, 1))
end

function train_model(m, data::DataFrame)
    opt = ADAGrad()
    Flux.train!(Flux.mse, params(m), data, opt)
end

# -------------- Apply corrections to magnetic moment ---------------

function apply_corrections(data::DataFrame, path)
    corr = CSV.File(path, header=1) |> DataFrame!
    for i in 1:nrow(corr)
        data[corr[i, :Index], :Moment] += corr[i, "Offset (emu)"]*1e6
    end
    return data
end

""" Filter out bad points from a hysteresis loop measurement. """
function heal_hyst(data::DataFrame; min_field=0.1, σ_threshold=0.1)
    transform!(
        groupby(data, :Field),
        [:Field, :Moment] =>
        (function (field, moment)
         σ = abs(std(moment)/mean(moment))
         if σ > σ_threshold
             println("Problem at B=$B: σ=$σ, ̄m=$(mean(d.Moment))")
             δ = abs.(moment .- mean(moment))
             bad = findall(δ .== maximum(δ))
             moment[bad] = mean(moment[Not(bad)])
             println(δ)
             println(moment)
         end
         return moment
         end) => :Moment)

end
# -------------- Analyze fitted magnetic moment data ---------------

function fit_hyst(data::DataFrame, filter=d->d.<-1.0)
    m = lm(@formula(Moment ~ Field), data[filter(data[!, :Field]), :])
    data.MomentFerro = data.Moment - GLM.coef(m)[2]*data.Field
    return data
end

function fit_hyst2_model(H,(Xdia, Ms, Mr, Hc))
    max_idx = findall(H .== maximum(H))[1]
    min_idx = findall(H .== minimum(H))[1]
    #(Xdia, Ms, Mr, Hc) = p
    idx1 = max_idx < min_idx ? max_idx : min_idx
    idx2 = max_idx < min_idx ? min_idx : max_idx
    r1 = 1:idx1
    r2 = idx1+1:idx2
    r3 = idx2+1:length(H)

    M = Xdia .* H
    M[r1] += (Ms*2/π) .* atan.((H[r1] ./ Hc) .* tan(π*Mr/(2*Ms)))
    M[r2] += (Ms*2/π) .* atan.((H[r2] .+ Hc) ./ Hc) .* tan(π*Mr/(2*Ms))
    M[r3] += (Ms*2/π) .* atan.((H[r3] .- Hc) ./ Hc) .* tan(π*Mr/(2*Ms))
    return M
end

function fit_hyst2_model2(H,(Xdia, Ms, Mr, Hc))
    max_idx = findall(H .== maximum(H))[1]
    min_idx = findall(H .== minimum(H))[1]
    #(Xdia, Ms, Mr, Hc) = p
    idx1 = max_idx < min_idx ? max_idx : min_idx
    idx2 = max_idx < min_idx ? min_idx : max_idx
    r1 = 1:idx1
    r2 = idx1+1:idx2
    r3 = idx2+1:length(H)

    M = Xdia .* H
    M[r1] += (Ms*2/π) .* tanh.((H[r1] ./ Hc) .* atanh(π*Mr/(2*Ms)))
    M[r2] += (Ms*2/π) .* tanh.((H[r2] .+ Hc) ./ Hc) .* atanh(π*Mr/(2*Ms))
    M[r3] += (Ms*2/π) .* tanh.((H[r3] .- Hc) ./ Hc) .* atanh(π*Mr/(2*Ms))
    return M
end


function fit_hyst2(data::DataFrame; model=fit_hyst2_model, p0=[-1.0, 1.0, 1.0, 1.0])
    return coef(curve_fit(model, data.Field, data.Moment, p0))
end

""" Estimate the diamagnetic susceptibility by fitting a line to parts of a hysteresis loop. """
function fit_dia(data::DataFrame, filter=d->d.<-1.0)
    m = lm(@formula(Moment ~ Field), data[filter(data[!, :Field]), :])
    return GLM.coef(m)[2]
end

""" Estimate the coercive field B_c from a hysteresis loop. """
function find_Bc(data::DataFrame)
    H = data.Field
    M = data.MomentFerro
    max_idx = findall(H .== maximum(H))[1]
    min_idx = findall(H .== minimum(H))[1]
    r1 = max_idx+1:min_idx
    r2 = min_idx+1:length(H)

    ps = sort(abs.(M[r1]))[1:2]
    p1 = findall(abs.(M[r1]) .== ps[1])[1] + max_idx
    p2 = findall(abs.(M[r1]) .== ps[2])[1] + max_idx
    Bc1 = coef(curve_fit((x,p)->p[1] .+ x.*p[2], [M[p1], M[p2]], [H[p1], H[p2]], [0.0, 1.0]))[1]

    ps = sort(abs.(M[r2]))[1:2]
    p1 = findall(abs.(M[r2]) .== ps[1])[1] + min_idx
    p2 = findall(abs.(M[r2]) .== ps[2])[1] + min_idx
    Bc2 = coef(curve_fit((x,p)->p[1] .+ x.*p[2], [M[p1], M[p2]], [H[p1], H[p2]], [0.0, 1.0]))[1]

    return mean(abs.([Bc1, Bc2]))
end

end
