import streamlit as st
import pandas as pd
import joblib

data = joblib.load("house_pipelines.pkl")

model = data["model"]
scaler = data["scaler"]
columns = data["columns"]
num_cols = data["num_cols"]

st.title("🏡 House Price Prediction")

# 🔢 Numeric inputs
OverallQual = st.number_input("Overall Quality", 1, 10, 5)
YearBuilt = st.number_input("Year Built", 1900, 2025, 2000)
YearRemodAdd = st.number_input("Year Remodel", 1900, 2025, 2005)
TotalBsmtSF = st.number_input("Basement Area", 0, 5000, 800)
FirstFlrSF = st.number_input("1st Floor Area", 0, 5000, 1000)
GrLivArea = st.number_input("Living Area", 0, 6000, 1500)
FullBath = st.slider("Full Bathrooms", 0, 4, 2)
TotRmsAbvGrd = st.slider("Total Rooms", 1, 15, 6)
GarageCars = st.slider("Garage Cars", 0, 4, 2)
GarageArea = st.number_input("Garage Area", 0, 2000, 400)

# 🔘 Categorical inputs
MSZoning = st.selectbox("Zoning", ["RL", "RM", "FV", "RH", "C (all)"])
Utilities = st.selectbox("Utilities", ["AllPub", "NoSeWa"])
BldgType = st.selectbox("Building Type", ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"])
Heating = st.selectbox("Heating", ["GasA", "GasW", "Grav", "Wall", "Floor", "OthW"])
KitchenQual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"])
SaleCondition = st.selectbox("Sale Condition", ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"])
LandSlope = st.selectbox("Land Slope", ["Gtl", "Mod", "Sev"])

# 🚀 Prediction
if st.button("Predict Price"):

    input_dict = {
        "OverallQual": OverallQual,
        "YearBuilt": YearBuilt,
        "YearRemodAdd": YearRemodAdd,
        "TotalBsmtSF": TotalBsmtSF,
        "1stFlrSF": FirstFlrSF,
        "GrLivArea": GrLivArea,
        "FullBath": FullBath,
        "TotRmsAbvGrd": TotRmsAbvGrd,
        "GarageCars": GarageCars,
        "GarageArea": GarageArea,

        "MSZoning": MSZoning,
        "Utilities": Utilities,
        "BldgType": BldgType,
        "Heating": Heating,
        "KitchenQual": KitchenQual,
        "SaleCondition": SaleCondition,
        "LandSlope": LandSlope
    }

    input_df = pd.DataFrame([input_dict])

    # 🔥 convert to 41 features
    input_df = pd.get_dummies(input_df)

    # match training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # scale only numeric
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prediction = model.predict(input_df)[0]

    st.success(f"🏡 Predicted Price: ${int(prediction):,}")