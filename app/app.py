
# V8 Compact Layout Upgrade Notes / Starter app structure
# Replace sections in app.py with this layout philosophy

LAYOUT = '''
ROW 1: Compact Hero (120-150px)
ROW 2: Horizontal Filter Bar (No Sidebar)
ROW 3: Executive Briefing | Priority Actions
ROW 4: 6 KPI Cards (2x3)
ROW 5: Risk Drivers | Regional Ranking
ROW 6: Customer Intelligence (2x2 grid)
ROW 7: Exposure Analysis
ROW 8: Governance Footer
'''

CHANGES = [
    "Remove sidebar and use st.columns filter bar",
    "Reduce hero height by 40%",
    "Use 6 KPI cards instead of 4",
    "Replace risk driver table with horizontal bar chart",
    "Replace duplicate regional chart with ranking table",
    "Add Gender and High Value customer intelligence cards",
    "Replace scatter-only exposure section with exposure scorecards",
    "Compact action center with Priority / Impact / Exposure",
    "Reduce vertical spacing between sections",
]
