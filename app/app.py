import os
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="ECB Banking Risk Intelligence Platform", layout="wide")

PRIMARY="#0F172A"; SECONDARY="#1E293B"; ACCENT="#334155"; BG="#F8FAFC"

st.markdown(f"""
<style>
.stApp {{background:{BG};}}
.kpi{{background:white;padding:1rem;border-radius:12px;border-left:5px solid {PRIMARY};}}
.big{{font-size:2rem;font-weight:700;color:{PRIMARY};}}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/European_Bank.csv")
    drop_cols=[c for c in ["Surname","CustomerId","Year"] if c in df.columns]
    df=df.drop(columns=drop_cols)
    df["AgeGroup"]=pd.cut(df["Age"],[0,30,45,60,120],labels=["<30","30-45","46-60","60+"])
    q75=df["Balance"].quantile(.75)
    df["IsHighValue"]=(df["Balance"]>=q75).astype(int)
    score=(df["IsActiveMember"].eq(0)*40 +
           (df["Age"]>45)*25 +
           (df["NumOfProducts"]>=3)*20 +
           df["IsHighValue"]*15)
    df["RiskScore"]=score
    df["RiskLevel"]=pd.cut(score,[-1,25,50,75,100],labels=["Low","Medium","High","Critical"])
    return df

df=load_data()

st.title("🏦 ECB Banking Risk Intelligence Platform")
st.caption("Executive Decision Support for Customer Attrition Risk")

with st.sidebar:
    st.header("Portfolio Filters")
    geo=st.multiselect("Country", sorted(df["Geography"].unique()), default=list(df["Geography"].unique()))
    risk=st.multiselect("Risk Level", list(df["RiskLevel"].cat.categories), default=list(df["RiskLevel"].cat.categories))

f=df[df["Geography"].isin(geo) & df["RiskLevel"].isin(risk)]

churn=f["Exited"].mean()*100
lost=int(f["Exited"].sum())
exposure=f.loc[f["Exited"]==1,"Balance"].sum()/1e6
critical=(f["RiskLevel"]=="Critical").sum()

st.subheader("Executive Risk Briefing")
top_geo=(f.groupby("Geography")["Exited"].mean()*100).sort_values(ascending=False)
top_geo_name=top_geo.index[0] if len(top_geo) else "N/A"

c1,c2,c3,c4=st.columns(4)
for c,title,val in [
    (c1,"Churn Rate",f"{churn:.1f}%"),
    (c2,"Customers Lost",f"{lost:,}"),
    (c3,"Balance Exposure",f"£{exposure:.1f}M"),
    (c4,"Critical Customers",f"{critical:,}")
]:
    with c:
        st.markdown(f"<div class='kpi'><div>{title}</div><div class='big'>{val}</div></div>", unsafe_allow_html=True)

st.warning(f"Highest risk geography: {top_geo_name}. Prioritize retention activity in this region.")

tab1,tab2,tab3,tab4,tab5=st.tabs([
    "Risk Segmentation","Geographic Intelligence","Customer Intelligence",
    "Portfolio Exposure","Governance"
])

with tab1:
    seg=f["RiskLevel"].value_counts().reset_index()
    seg.columns=["RiskLevel","Customers"]
    st.plotly_chart(px.pie(seg,names="RiskLevel",values="Customers",title="Risk Segmentation"),use_container_width=True)

with tab2:
    geo_df=(f.groupby("Geography")["Exited"].mean()*100).reset_index()
    geo_df.columns=["Geography","ChurnRate"]
    st.plotly_chart(px.bar(geo_df,x="Geography",y="ChurnRate",color="ChurnRate",
                           title="Country Risk Ranking"),use_container_width=True)

with tab3:
    age=(f.groupby("AgeGroup")["Exited"].mean()*100).reset_index()
    st.plotly_chart(px.bar(age,x="AgeGroup",y="Exited",title="Age Risk Analysis"),
                    use_container_width=True)

with tab4:
    st.plotly_chart(px.scatter(f.sample(min(len(f),2000)),
                               x="Balance",y="EstimatedSalary",
                               color="RiskLevel",
                               title="Portfolio Exposure Map"),
                    use_container_width=True)

    recs=[]
    if top_geo_name!="N/A":
        recs.append(f"Launch targeted retention campaign in {top_geo_name}")
    if critical>0:
        recs.append("Assign relationship managers to critical-risk customers")
    st.subheader("Action Center")
    for r in recs:
        st.success(r)

with tab5:
    st.metric("Records", len(f))
    st.write("Dataset: European_Bank.csv")
    st.write("Dashboard Version: 2.0 ECB Edition")
    st.write("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M"))

csv=f.to_csv(index=False).encode()
st.download_button("Download Filtered Data", csv, "filtered_data.csv","text/csv")
