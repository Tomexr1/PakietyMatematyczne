using DataFrames: DataFrame, missing, eltype
using CSV
using XGBoost
using PyCall
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split
# using ScikitLearn.Preprocessing: Label
# @sk_import preprocessing: LabelEncoder
@sk_import preprocessing: (LabelBinarizer, StandardScaler)
# @sk_import feature_extraction: DictVectorizer
# using ScikitLearn.feature_extraction: DictVectorizer
# using ScikitLearn: fit!


weather_data = DataFrame(CSV.File("weather.csv", normalizenames=true, delim=","))
# describe(weather_data)
unique(weather_data[!, :])
# println(names(weather_data))
# first(weather_data)
select!(weather_data, Not("RISK_MM"))
# println(names(weather_data))
# print(describe(weather_data))
# df_dict = Dict(pairs(eachcol(weather_data)))


# df_dict = Dict(pairs(weather_data.RainTomorrow))
# dv = DictVectorizer()
# df_encoded = fit_transform!(dv, df_dict)
# df_encoded[0][0]

# categorical_mask = (df.dtypes == object)
# categorical_columns = 
# eltype.(eachcol(weather_data))
mapper = DataFrameMapper([([:WindGustDir,:WindGustSpeed,:WindDir9am,:WindDir3pm,:RainToday,:RainTomorrow], LabelBinarizer()), ([:MinTemp,:MaxTemp,:Rainfall,:Evaporation,:Sunshine,:WindSpeed9am,:WindSpeed3pm,:Humidity9am,:Humidity3pm,:Pressure9am,:Pressure3pm,:Cloud9am,:Cloud3pm,:Temp9am,:Temp3pm], nothing)])

mapper2 = DataFrameMapper([(:MinTemp, nothing),
(:MaxTemp, nothing),
(:Rainfall, nothing),
(:Evaporation, nothing),
(:Sunshine, nothing),
(:WindSpeed9am, nothing),
(:WindSpeed3pm, nothing),
(:Humidity9am, nothing),
(:Humidity3pm, nothing),
(:Pressure9am, nothing),
(:Pressure3pm, nothing),
(:Cloud9am, nothing),
(:Cloud3pm, nothing),
(:Temp9am, nothing),
(:Temp3pm, nothing),
(:WindGustDir, LabelBinarizer()),
(:WindGustSpeed, nothing),
(:WindDir9am, LabelBinarizer()),
(:WindDir3pm, LabelBinarizer()),
(:WindSpeed9am, nothing),
(:WindSpeed3pm, nothing),
(:Humidity9am, nothing),
(:Humidity3pm, nothing),
(:Pressure9am, nothing),
(:Pressure3pm, nothing),
(:Cloud9am, nothing),
(:Cloud3pm, nothing),
(:Temp9am, nothing),
(:Temp3pm, nothing),
(:RainToday, LabelBinarizer()),
(:RainTomorrow, LabelBinarizer())]
)
# fit_transform!(mapper, weather_data)
fit_transform!(mapper2, copy(weather_data))
