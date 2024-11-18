import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

def app():
    st.title("CSV Data Visualization")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    filtered_df = pd.DataFrame()
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.dropna(subset=df.columns, inplace=True)

        st.subheader("Filtered Data")
        st.dataframe(df)

        st.subheader("Line Graph")
        x_column = st.selectbox("Select a column for the X-axis", df.columns)
        y_column = st.selectbox("Select a column for the Y-axis", df.columns)

        

        if (
            df[x_column].dtype == "float64"
            and pd.to_numeric(df[y_column], errors='coerce').notna().all()
        ):
            chart = alt.Chart(df).mark_line().encode(
                x=x_column,
                y=alt.Y(y_column, title=y_column),
            ).properties(
                width=600,
                height=400,
                title=f"Line Chart for {y_column} over {x_column}",
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Selected columns cannot be visualized as a line chart due to mixed data types.")

            