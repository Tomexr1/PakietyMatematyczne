using DataFrames
using CSV
using XGBoost
using XGBoost: predict
using PyCall
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split, cross_val_score
using Impute: Impute
using Random
using Plots
using MLBase
@sk_import preprocessing: (LabelBinarizer, StandardScaler)

weather_data = DataFrame(CSV.File("weather.csv", normalizenames=true, delim=",", missingstring="NA"))
unique(weather_data[!, :])
select!(weather_data, Not("RISK_MM"))
Impute.srs!(weather_data; rng=MersenneTwister(1234))

mapper2 = DataFrameMapper([([:MinTemp], StandardScaler()),
([:MaxTemp], StandardScaler()),
([:Rainfall], StandardScaler()),
([:Evaporation], StandardScaler()),
([:Sunshine], StandardScaler()),
([:WindSpeed9am], StandardScaler()),
([:WindSpeed3pm], StandardScaler()),
([:Humidity9am], StandardScaler()),
([:Humidity3pm], StandardScaler()),
([:Pressure9am], StandardScaler()),
([:Pressure3pm], StandardScaler()),
([:Cloud9am], StandardScaler()),
([:Cloud3pm], StandardScaler()),
([:Temp9am], StandardScaler()),
([:Temp3pm], StandardScaler()),
(:WindGustDir, LabelBinarizer()),
([:WindGustSpeed], StandardScaler()),
(:WindDir9am, LabelBinarizer()),
(:WindDir3pm, LabelBinarizer()),
([:WindSpeed9am], StandardScaler()),
([:WindSpeed3pm], StandardScaler()),
([:Humidity9am], StandardScaler()),
([:Humidity3pm], StandardScaler()),
([:Pressure9am], StandardScaler()),
([:Pressure3pm], StandardScaler()),
([:Cloud9am], StandardScaler()),
([:Cloud3pm], StandardScaler()),
([:Temp9am], StandardScaler()),
([:Temp3pm], StandardScaler()),
(:RainToday, LabelBinarizer()),
(:RainTomorrow, LabelBinarizer())]
)


data_matrix = fit_transform!(mapper2, copy(weather_data))

X, y = data_matrix[:,1:end-1], data_matrix[:,end]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
xg_cl = xgboost((X_train, y_train), num_round=5, max_depth=6, objective="binary:logistic")
preds = predict(xg_cl, X_test)
acuracy = sum(round.(preds) .== y_test) / length(y_test)

imp = DataFrame(importancetable(xg_cl))

dtrain = DMatrix(X_train, label=y_train)
boost = xgboost(dtrain, eta = 1, objective = "binary:logistic", max_depth=4, tree_method="exact")
prediction = XGBoost.predict(boost, X_test)
prediction_rounded = Array{Int64, 1}(map(val -> round(val), prediction))
accuracy = sum(prediction_rounded .== y_test) / length(y_test)
MLBase.confusmat(2, Array{Int64, 1}(y_test .+ 1), Array{Int64, 1}(prediction_rounded .= 1))
