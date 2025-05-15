import joblib
import streamlit as st # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("E Commerce Dataset.csv")
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors='ignore')
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
df["CityTier"] = df["CityTier"].map({1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}).astype("category")
df["CashbackAmount"] = df["CashbackAmount"].replace({"\\$": "", ",": ""}, regex=True)
df["CashbackAmount"] = pd.to_numeric(df["CashbackAmount"], errors='coerce').fillna(0).astype(float)
df["OrderCount"] = pd.to_numeric(df["OrderCount"], errors='coerce').fillna(0).astype(int)

# Load model and scaler
model = joblib.load("retrain_churn_model.pkl")
scaler = joblib.load("re_scaler.pkl")

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")
st.title("🛍️ E-Commerce Dashboard & Churn Prediction")

page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔮 Churn Prediction"])

if page == "📊 Dashboard":
    st.header("📈 Business Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", df.shape[0])
    col2.metric("Total Orders", int(df["OrderCount"].sum()))
    col3.metric("Avg Cashback", f"${df['CashbackAmount'].mean():.2f}")

    st.subheader("Churn Distribution")
    churn_counts = df['Churn'].value_counts()
    fig1, ax1 = plt.subplots()
    st.subheader("Churn Distribution")
    churn_counts = df['Churn'].value_counts()
    labels = churn_counts.index.map({0: 'No', 1: 'Yes'}).tolist()
    fig1, ax1 = plt.subplots()
    ax1.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("Orders by City Tier")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df, x="CityTier", y="OrderCount", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Cashback by Churn")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x="Churn", y="CashbackAmount", ax=ax3)
    st.pyplot(fig3)

    st.subheader("📄 Raw Dataset")
    st.dataframe(df)

elif page == "🔮 Churn Prediction":
    st.header("📌 Predict Customer Churn")

    default_features = [col for col in df.columns if col != "Churn"]
    selected_features = st.multiselect("Select Features for Prediction", options=default_features, default=default_features)

    if selected_features:
        X = df[selected_features]
        X = pd.get_dummies(X, drop_first=True)
        X.fillna(0, inplace=True)

        # Scale input features using loaded scaler
        X_scaled = scaler.transform(X)

        # Predict churn using loaded model
        y_pred = model.predict(X_scaled)

        st.subheader("🔍 Predicted Churn for Selected Features")
        result_df = df[selected_features].copy()
        result_df["Churn_Predicted"] = y_pred
        st.dataframe(result_df)

        st.download_button("📥 Download Predictions as CSV", result_df.to_csv(index=False), "churn_predictions.csv", "text/csv")
    else:
        st.warning("Please select at least one feature to continue.")
