using PyCall
# xgb = pyimport("xgboost")
using DataFrames
using CSV
using ScikitLearn
using Random
using XGBoost
using MLJ
# XGBoostRegressor = @load XGBoostRegressor pkg=XGBoost
using RDatasets

function splitdf(df, pct)
    """Split a DataFrame into two DataFrames, the first containing `pct` of the rows, and the second containing the remaining rows."""
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end


weather_data = DataFrame(CSV.File("bazadanychcsv.csv", normalizenames=true, delim=";"))
names(weather_data)
# iris_df = DataFrame(dataset("datasets", "iris"))

# X = convert(Array, weather_data[[:Rok, :Miesiąc, :Dzień, :zachmurzenie, :cisnienie_pary_wodnej, :wilgotnosc_wzgl, :cisnienie_na_poziomie_stacji, :cisnienie_na_poziomie_morza, :suma_opadu_dzien, :suma_opadu_noc]])
# y = convert(Array, weather_data[:srednia_temp])
# unikalne stacje
# stations = unique(weather_data[!, :Nazwa_stacji])
# weather_data[19300:end, :]



test, train = splitdf(weather_data, 0.2)
y_train, y_test = train[:, :srednia_temp], test[:, :srednia_temp]
# X_train = train[:, [:Rok, :Miesiąc, :Dzień, :zachmurzenie, :cisnienie_pary_wodnej, :wilgotnosc_wzgl, :cisnienie_na_poziomie_stacji, :cisnienie_na_poziomie_morza, :suma_opadu_dzien, :suma_opadu_noc]]
# X_test = test[:, [:Rok, :Miesiąc, :Dzień, :zachmurzenie, :cisnienie_pary_wodnej, :wilgotnosc_wzgl, :cisnienie_na_poziomie_stacji, :cisnienie_na_poziomie_morza, :suma_opadu_dzien, :suma_opadu_noc]]
X_train = train[:, [:wilgotnosc_wzgl, :cisnienie_na_poziomie_stacji, :cisnienie_na_poziomie_morza, :suma_opadu_dzien, :suma_opadu_noc]]
X_test = test[:, [:Rok, :Miesiąc, :Dzień, :zachmurzenie, :cisnienie_pary_wodnej, :wilgotnosc_wzgl, :cisnienie_na_poziomie_stacji, :cisnienie_na_poziomie_morza, :suma_opadu_dzien, :suma_opadu_noc]]

# convert X_train to AbstractMatrix
# X_train = convert(dataset, X_train)
# X_test = convert(Array, X_test)
# y_train = convert(Vector, y_train)
# DMatrix(X_train, label=y_train)
# @load XGBoostRegressor pkg=XGBoost

# X = weather_data[:, [:Rok, :Miesiąc, :Dzień, :zachmurzenie, :cisnienie_pary_wodnej, :wilgotnosc_wzgl, :cisnienie_na_poziomie_stacji, :cisnienie_na_poziomie_morza, :suma_opadu_dzien, :suma_opadu_noc]]
# y = weather_data[:, :srednia_temp]
# (Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.8, rng=123, multi=true)
# xgb_re = XGBoostRegressor()
# mach = machine(xgb_re, Xtrain, ytrain)




