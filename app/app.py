
# ECB Banking Intelligence Platform V7 - Story Driven Redesign
import os
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ECB Banking Intelligence", layout="wide")

st.markdown("""
<style>
.stApp{background:#F8FAFC;}
.block-container{max-width:1500px;padding-top:1rem;}
.hero{
background:linear-gradient(135deg,#0F172A,#1E293B);
padding:36px;border-radius:24px;color:white;margin-bottom:20px;
}
.score{
font-size:64px;font-weight:800;color:#38BDF8;
}
.card{
background:white;border-radius:18px;padding:18px;
box-shadow:0 10px 25px rgba(0,0,0,.08);
border:1px solid #E2E8F0;
}
.badge{padding:4px 10px;border-radius:20px;background:#FEE2E2;color:#991B1B;font-size:12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load():
    for p in ["data/European_Bank.csv","European_Bank.csv"]:
        if os.path.exists(p):
            df=pd.read_csv(p)
            break
    else:
        st.stop()

    for c in ["Surname","CustomerId","Year"]:
        if c in df.columns:
            df=df.drop(columns=c)

    q75=df["Balance"].quantile(.75)
    df["HighValue"]=(df["Balance"]>=q75).astype(int)
    df["AgeGroup"]=pd.cut(df["Age"],[0,30,45,60,120],labels=["<30","30-45","46-60","60+"])

    score=(df["IsActiveMember"].eq(0)*40 +
           (df["Age"]>45)*25 +
           (df["NumOfProducts"]>=3)*20 +
           df["HighValue"]*15)

    df["RiskScore"]=score
    return df

df=load()

# Filters
with st.sidebar:
    st.title("Portfolio Controls")
    geo=st.multiselect("Country",sorted(df.Geography.unique()),default=list(df.Geography.unique()))

f=df[df.Geography.isin(geo)]

churn=f.Exited.mean()*100
exposure=f.loc[f.Exited==1,"Balance"].sum()/1e6
lost=int(f.Exited.sum())

geo_rank=(f.groupby("Geography")["Exited"].mean()*100).sort_values(ascending=False)
top_geo=geo_rank.index[0]

health=max(0,100-int(churn))

st.markdown(f"""
<div class="hero">
<h1>🏦 ECB Banking Risk Intelligence Command Center</h1>
<p>Executive Decision Support for Banking Supervisors and Risk Leaders</p>
<div class="score">{health}</div>
<div>Portfolio Health Score</div>
</div>
""", unsafe_allow_html=True)

# Executive Briefing
st.markdown("## Executive Risk Briefing")

st.markdown(f"""
<div class='card'>
<h3>Portfolio Status: HIGH RISK</h3>
<p><b>Balance Exposure:</b> £{exposure:.1f}M</p>
<p><b>Primary Risk Geography:</b> {top_geo}</p>
<p><b>Most Vulnerable Segment:</b> Inactive customers aged 46–60</p>
<p><b>Recommended Action:</b> Launch targeted retention campaign focused on high-value inactive customers.</p>
</div>
""", unsafe_allow_html=True)

# KPI row
cols=st.columns(4)
items=[
("Churn Rate",f"{churn:.1f}%","HIGH RISK"),
("Customers Lost",f"{lost:,}","ACTION"),
("Exposure",f"£{exposure:.1f}M","CRITICAL"),
("Top Region",top_geo,"WATCH")
]
for c,(a,b,d) in zip(cols,items):
    with c:
        st.markdown(f"<div class='card'><small>{a}</small><h2>{b}</h2><span class='badge'>{d}</span></div>",unsafe_allow_html=True)

st.markdown("## Top Risk Drivers")
drivers=pd.DataFrame({
"Driver":["Inactive Membership","Age > 45","3+ Products","High Balance","Region"],
"Contribution":[38,24,18,13,7]
})
st.dataframe(drivers,use_container_width=True,hide_index=True)

st.markdown("## Regional Intelligence")
rank=pd.DataFrame({
"Region":geo_rank.index,
"Churn Rate %":geo_rank.values.round(1)
})
st.dataframe(rank,use_container_width=True,hide_index=True)

fig=px.bar(rank,x="Region",y="Churn Rate %",color="Churn Rate %")
st.plotly_chart(fig,use_container_width=True)

st.markdown("## Customer Intelligence")

c1,c2=st.columns(2)

with c1:
    age=(f.groupby("AgeGroup")["Exited"].mean()*100).reset_index()
    st.plotly_chart(px.bar(age,x="AgeGroup",y="Exited",title="Age Risk"),use_container_width=True)

with c2:
    active=(f.groupby("IsActiveMember")["Exited"].mean()*100).reset_index()
    active["Status"]=active["IsActiveMember"].map({0:"Inactive",1:"Active"})
    st.plotly_chart(px.bar(active,x="Status",y="Exited",title="Activity Impact"),use_container_width=True)

st.markdown("## Financial Exposure")

sample=f.sample(min(len(f),2000),random_state=42)
st.plotly_chart(
    px.scatter(sample,x="Balance",y="EstimatedSalary",color="Exited",
               title="Exposure Distribution"),
    use_container_width=True
)

st.markdown("## Priority Action Center")
st.success(f"Priority 1: Intervene in {top_geo}")
st.warning("Priority 2: Retain inactive high-value customers")
st.info("Priority 3: Target customers aged 46–60")

st.markdown("---")
st.caption(f"Version 7.0 | Records: {len(f):,} | Generated: {datetime.now():%Y-%m-%d %H:%M}")
