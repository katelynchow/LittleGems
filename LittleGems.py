import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import calendar 

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
    'Very_High': '#CCCCFF', 
    'High': '#FFB6C1',
    'Higher': '#90EE90', # light green
    'Average': '#ADD8E6', # light blue 
    'Low': '#C8D9F0'
}
df['Color'] = df['TrafficGroupSimplified'].map(color_map)

years = df[date_col].dt.year.unique()

html_full = ""

for year in sorted(years):
    months = df[df[date_col].dt.year == year][date_col].dt.month.unique()
    for month in sorted(months):
        cal = calendar.Calendar()
        html = '<table border="1" style="border-collapse: collapse; text-align:center; margin-bottom:20px;">'
        html += f'<tr><th colspan="7">{calendar.month_name[month]} {year}</th></tr>'
        html += '<tr>' + ''.join(f'<th>{day}</th>' for day in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']) + '</tr>'

        for week in cal.monthdatescalendar(year, month): # I mean this is just a loop over the calendar year 
            html += '<tr>'
            for day in week:
                if day.month == month:
                    # get color for this date
                    row = df[df[date_col].dt.date == day]
                    if not row.empty:
                        color = row['Color'].values[0]
                    else:
                        color = 'white'  # empty day
                    html += f'<td style="background-color:{color}; width:40px; height:40px;">{day.day}</td>'
                else:
                    html += '<td></td>'  
            html += '</tr>'
        html += '</table>'

        html_full += html

with open('traffic_calendar.html', 'w') as f:
    f.write(html_full)

print("HTML calendar created: traffic_calendar.html")