using DataFrames
using CSV
using XGBoost
using XGBoost: predict
using PyCall
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split
using Impute: Impute
using Random
# using ScikitLearn.Preprocessing: Label
# @sk_import preprocessing: LabelEncoder
@sk_import preprocessing: (LabelBinarizer, StandardScaler, OneHotEncoder)
# @sk_import feature_extraction: DictVectorizer
# using ScikitLearn.feature_extraction: DictVectorizer
# using ScikitLearn: fit!


weather_data = DataFrame(CSV.File("weather.csv", normalizenames=true, delim=",", missingstring="NA"))
unique(weather_data[!, :])
select!(weather_data, Not("RISK_MM"))
Impute.srs!(weather_data; rng=MersenneTwister(1234))
# print(describe(weather_data))
# df_dict = Dict(pairs(eachcol(weather_data)))


# df_dict = Dict(pairs(weather_data.RainTomorrow))
# dv = DictVectorizer()
# df_encoded = fit_transform!(dv, df_dict)
# df_encoded[0][0]

# categorical_mask = (df.dtypes == object)
# categorical_columns = 
# eltype.(eachcol(weather_data))
# mapper = DataFrameMapper([([:WindGustDir,:WindGustSpeed,:WindDir9am,:WindDir3pm,:RainToday,:RainTomorrow], LabelBinarizer()), ([:MinTemp,:MaxTemp,:Rainfall,:Evaporation,:Sunshine,:WindSpeed9am,:WindSpeed3pm,:Humidity9am,:Humidity3pm,:Pressure9am,:Pressure3pm,:Cloud9am,:Cloud3pm,:Temp9am,:Temp3pm], nothing)])

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
([:WindGustDir], OneHotEncoder(sparse=false)),
([:WindGustSpeed], StandardScaler()),
([:WindDir9am], OneHotEncoder(sparse=false)),
([:WindDir3pm], OneHotEncoder(sparse=false)),
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
# fit_transform!(mapper, weather_data)
data_matrix = fit_transform!(mapper2, copy(weather_data))

function partitionTrainTest(data, at = 0.7)
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
end

# train,test = partitionTrainTest(weather_data, 0.7)
X, y = data_matrix[:,1:end-1], data_matrix[:,end]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
xg_cl = xgboost((X_train, y_train), num_round=5, max_depth=6, objective="binary:logistic")
preds = predict(xg_cl, X_test)
acuracy = sum(round.(preds) .== y_test) / length(y_test)