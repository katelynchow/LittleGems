import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import calendar 
from datetime import date, timedelta, datetime

# Load the file with the new name
fname = "Waffles.csv"
df = pd.read_csv(fname)

date_col = "Date"  # Change if the column is named differently
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# CLEAN DATA: Strip currency symbols and commas then convert to float
for col in ['Net Sales', 'Orders', 'Discounts']:
    df[col] = (
        df[col].astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.replace('(', '-', regex=False)  # handles possible negatives in parens
        .str.replace(')', '', regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute isWeekend
df['isWeekend'] = df[date_col].dt.dayofweek >= 5

# Prepare features for clustering
features = df[['Net Sales', 'Orders', 'Discounts', 'isWeekend']].copy()
features['isWeekend'] = features['isWeekend'].astype(int)

# Drop rows with missing values in any feature
features = features.dropna()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# -- Commented out code that defines the optimal amounts of clusters -- 
# K_range = range(2, 10)
# inertia = []
# silhouette_scores = []

# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
#     labels = kmeans.fit_predict(X_scaled)
#     inertia.append(kmeans.inertia_)
#     silhouette_scores.append(silhouette_score(X_scaled, labels))

# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(K_range, inertia, marker='o')
# plt.title('Elbow Method for Optimal K')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Inertia')
# plt.grid(True)

# plt.subplot(1,2,2)
# plt.plot(K_range, silhouette_scores, marker='o', color='orange')
# plt.title('Silhouette Score for Optimal K')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Average Silhouette Score')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# for k, inert, sil in zip(K_range, inertia, silhouette_scores):
#     print(f"K={k}: Inertia={inert:.2f}, Silhouette Score={sil:.3f}") 
# -----------

# K-Means clustering 

kmeans = KMeans(n_clusters=5, random_state=42)

df.loc[features.index, 'TrafficGroup'] = kmeans.fit_predict(X_scaled)
# Based on kmean library, it utilizes the "Traffic group" (which is legit just the features Net sales, orders, discounts, isWeekend)
# to decide on what clusters each datapoint goes into. For example, a "high" may be a day with high sales (duh) but notably on a WEEKDAY whereas
# a mid can be a high sale day with a lot of discounts and a weekend

df['TrafficGroupSimplified'] = df['TrafficGroup'].replace({
    0: 'Very_High', # combined groups 0 and 4 because their average sales were similar enough
    4: 'High',
    2: 'Higher',
    1: 'Low',
    3: 'Average' # look at printout for mean sale count to see why I clustered like this.
})

# Plot grouped net sales
plt.figure(figsize=(10,5))
for label in df['TrafficGroupSimplified'].unique():
    group = df[df['TrafficGroupSimplified'] == label]
    plt.scatter(group[date_col], group['Net Sales'], label=label, s=10)
plt.legend()
plt.title("Net Sales by Simplified Foot Traffic Groups")
plt.xlabel("Date")
plt.ylabel("Net Sales")
plt.tight_layout()
plt.show()

# Output stats
print(df.groupby('TrafficGroup')[['Net Sales', 'Orders', 'Discounts']].describe())

# Note:
# High: Days with genuinely high revenue (sales + orders)
# Medium: Days with moderate sales and orders
# Low: Days with low revenue
# Discounts and weekend effects are secondary factors that influence cluster assignment but don’t dominate if sales are already very high. 
# Honestly should graph the discount vs net sales for the high clusters 

# -- HTML calendar script --

color_map = {
    'Very_High': '#B6A2D8', #purple
    'High': '#E8A1A5', #red
    'Average': '#D7E9CB', # light green
    'Lower': '#9DBCF2', # light blue 
    'Low': '#C7DAF5', #ice blue
}
df['Color'] = df['TrafficGroupSimplified'].map(color_map)

df['weekday'] = df[date_col].dt.weekday
df['week_of_month'] = df[date_col].apply(lambda d: (d.day - 1) // 7 + 1)
df['month'] = df[date_col].dt.month
df['year'] = df[date_col].dt.year

# -----------------------------
# CUSTOM LABELS FROM 2024 DATA
# (You can edit these for 2024)
# -----------------------------
df['Label'] = ""  # default blank

start = "2025-10-01"
end = "2026-12-31"
# Example events in 2025-2026 (modify these!). Please REMEMBER TO ADD A COMMA AFTER EACH ONE
custom_labels = {
    "2025-10-17": "Cal VS UNC Football",
    "2025-10-31": "Halloween",
    "2025-11-26": "No Instruction",
    "2025-11-27": "Thanksgiving",
    "2026-11-29": "Cal VS SMU Football",
    "2025-12-8": "RRR Week Begins",
    "2025-12-15": "Final Examination",
    "2025-12-20": "Winter Break Begins",
    "2025-12-24": "Christmas Eve",
    "2025-12-25": "Christmas",
    "2026-1-20": "Class Resumes",
    "2026-2-16": "MLK Day",
    "2026-3-17": "St. Patrick's Day",
    "2026-3-21": "Spring Break",
    "2026-3-30": "Class Resumes",
    "2026-4-20": "420",
    "2026-5-4": "RRR Week Begins",
    "2026-5-11": "Final Examination",
    "2026-5-16": "Commencement",
    "2026-5-17": "Summer Break Begins",
    "2026-5-26": "Summer Session A Begins",
    "2026-6-8": "Summer Session B Begins",
    "2026-8-15": "Move In",
    "2026-8-26": "Class Resumes",
    "2026-9-5": "Cal VS UCLA Football",
    "2026-9-19": "Cal VS Wagner Football",
    "2026-10-31": "Halloween",
    "2026-11-21": "Cal VS Stanford Football (Big Game)",
    "2026-11-26": "Thanksgiving",
    "2026-12-7": "RRR Week Begins",
    "2026-12-14": "Final Examination",
    "2026-12-19": "Winter Break Begins",
    "2026-12-24": "Christmas Eve",
    "2026-12-25": "Christmas",
}
# custom_labels.update(generate_taco_tuesday(start, end))
# for k,v in custom_labels.items():
#     df.loc[df[date_col] == k, "Label"] = v

# -----------------------------
# BUILD FUTURE DATE RANGE
# Oct 2025 → Dec 2026
# -----------------------------

future_dates = pd.date_range(start, end, freq='D')
future = pd.DataFrame({'Date': future_dates})

future['weekday'] = future['Date'].dt.weekday
future['week_of_month'] = future['Date'].apply(lambda d: (d.day - 1) // 7 + 1)
future['month'] = future['Date'].dt.month
future['year'] = future['Date'].dt.year

# -----------------------------
# PROJECTION:
# Match (month, weekday, week_of_month) with 2024 data
# -----------------------------
template = (
    df.sort_values(date_col)      # ensures consistent overwrite order
      .drop_duplicates(
          subset=['month', 'weekday', 'week_of_month'],
          keep='last'
      )
      .copy()
)

projection = future.merge(
    template[['month', 'weekday', 'week_of_month', 'TrafficGroupSimplified', 'Color', 'Label']],
    on=['month', 'weekday', 'week_of_month'],
    how='left'
)
projection['TextColor'] = None # default text color

for k, v in custom_labels.items():
    d = pd.to_datetime(k)
    projection.loc[projection['Date'] == d, "Label"] = v

    # if v == "Taco Tuesday":
    #     projection.loc[projection['Date'] == d, "Color"] = "#FFEAC4"

# -----------------------------
# HTML CALENDAR GENERATION
# -----------------------------
html_full = ""

# Calendar range:
start_year, start_month = 2025, 10   # October 2025
end_year, end_month = 2026, 12       # December 2026

for year in range(start_year, end_year + 1):

    if year == start_year:
        months = range(start_month, 13)
    elif year == end_year:
        months = range(1, end_month + 1)
    else:
        months = range(1, 13)

    for month in months:
        cal = calendar.Calendar()

        html = '''
        <table border="1" style="border-collapse: collapse; text-align:center;
                                margin-bottom:20px; width: 100%; font-family: Arial;">
        '''
        html += f'<tr><th colspan="7" style="padding:10px; font-size:20px;">{calendar.month_name[month]} {year}</th></tr>'
        html += '<tr>' + ''.join(
            f'<th style="padding:5px;">{d}</th>' 
            for d in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        ) + '</tr>'

        # generate weeks
        for week in cal.monthdatescalendar(year, month):
            html += "<tr>"

            for day in week:
                if day.month == month:

                    # look up projected data
                    row = projection[projection['Date'] == pd.to_datetime(day)]

                    if not row.empty:
                        color = row['Color'].values[0] if pd.notna(row['Color'].values[0]) else 'white'
                        label = row['Label'].values[0] if pd.notna(row['Label'].values[0]) else ""
                    else:
                        color = 'white'
                        label = ""

                    html += f'''
                    <td style="background-color:{color}; width:130px; height:90px; vertical-align:top;">
                        <div style="font-weight:bold; font-size:16px; margin-top:5px;">{day.day}</div>
                        <div style="font-size:12px; margin-top:5px;">{label}</div>
                    </td>
                    '''
                else:
                    html += "<td></td>"

            html += "</tr>"

        html += "</table>"
        html_full += html


with open("traffic_calendar_projection.html", "w") as f:
    f.write(html_full)

print("Projection calendar created: traffic_calendar_projection.html")

