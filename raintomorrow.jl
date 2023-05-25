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


# data_matrix = fit_transform!(mapper2, copy(weather_data))

# X, y = data_matrix[:,1:end-1], data_matrix[:,end]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# xg_cl = xgboost((X_train, y_train), num_round=5, max_depth=6, objective="binary:logistic")
# preds = predict(xg_cl, X_test)
# acuracy = sum(round.(preds) .== y_test) / length(y_test)

# imp = DataFrame(importancetable(xg_cl))

# dtrain = DMatrix(X_train, label=y_train)
# boost = xgboost(dtrain, eta = 1, objective = "binary:logistic", max_depth=4, tree_method="exact")
# prediction = XGBoost.predict(boost, X_test)
# prediction_rounded = Array{Int64, 1}(map(val -> round(val), prediction))
# accuracy = sum(prediction_rounded .== y_test) / length(y_test)
# MLBase.confusmat(2, Array{Int64, 1}(y_test .+ 1), Array{Int64, 1}(prediction_rounded .= 1))




# dump_model(xg_cl, "dump.raw.txt")

# param = ["max_depth" => 2,
#          "eta" => 1,
#          "objective" => "binary:logistic"]

# metrics = ["auc"]

# using ScikitLearn.CrossValidation: cross_val_score

# cv_scores = cv(X_train, 5, 4, label = y_train, param = param, metrics = metrics)

# imp = importancereport(xg_cl)
# trees(xg_cl)[1]

using MLJ

# XGBC = @load XGBoostClassifier pkg=XGBoost
# xgb_model = XGBC()

# mach = machine(xgb_model, X, y)

# cv=CV(nfolds=3)
# evaluate(xg_cl)
# evaluate!(xg_cl, resampling=cv, measure=12, verbosity=0)
# evaluate(xgb_model, X, y, resampling=cv, measure=l2, verbosity=0)

# evaluate(xgb_model, X, y, resampling=CV(shuffle=true),measures=[log_loss, accuracy],verbosity=0)
# y = convert(Vector{Bool}, y)
# train, test = partition(eachindex(y), 0.5, shuffle=true, rng=333)
# MLJ.fit!(mach, rows=train)
# ypred = MLJ.predict(mach, rows=test)
y, X =  MLJ.unpack(weather_data, ==(:RainTomorrow), rng=123);
(X_train, X_test), (y_train, y_test)  = partition((X, y), 0.6, multi=true,  rng=123);
pipe = (MLJ.FillImputer() |> MLJ.OneHotEncoder(drop_last = true))

mach_transf = machine(pipe, X_train) |> MLJ.fit!

mach_transf
X_train = MLJ.transform(mach_transf, X_train);
X_test = MLJ.transform(mach_transf, X_test);
schema(X_train)
ms = MLJ.models(matching(X_train, y_train))
base_model = @load XGBoostClassifier pkg=XGBoost

xgb_model = base_model(max_depth = 4, num_round = 200)
mach_xgb = machine(xgb_model, X_train, y_train)

MLJ.fit!(mach_xgb, verbosity=2)