# FOOD DELIVERY OPERATIONS ANALYTICS
# Python + Data Analytics Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TASK 109: Load and inspect dataset
df = pd.read_csv("08_food_delivery.csv")

print("First five records:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nDataset shape:")
print(df.shape)

# TASK 110: Handle missing delivery time
cancelled = df["Order_Status"].str.lower() == "cancelled"
delivered = df["Order_Status"].str.lower() == "delivered"

df.loc[cancelled, "Delivery_Time"] = np.nan

for city in df["City"].dropna().unique():
    mask = (df["City"] == city) & delivered
    median = df.loc[mask, "Delivery_Time"].median()
    df.loc[mask, "Delivery_Time"] = (
        df.loc[mask, "Delivery_Time"].fillna(median)
    )

# TASK 111: Remove duplicate orders
df = df.drop_duplicates(subset="Order_ID", keep="first")

# TASK 112: Total orders and status percentage
print("\nTotal Orders:", len(df))
print("\nOrder Status Percentage:")
print(df["Order_Status"].value_counts(normalize=True) * 100)

# TASK 113: Average delivery time by city
print("\nAverage Delivery Time by City:")
print(df.groupby("City")["Delivery_Time"].mean())

# TASK 114: Top five restaurants by order value
print("\nTop Five Restaurants:")
top5 = df.groupby("Restaurant")["Order_Value"].sum()
print(top5.sort_values(ascending=False).head(5))

# TASK 115: Average order value by restaurant
print("\nAverage Order Value:")
print(df.groupby("Restaurant")["Order_Value"].mean())

# TASK 116: Peak ordering hours
df["Order_Time"] = pd.to_datetime(df["Order_Time"], errors="coerce")
df["Order_Hour"] = df["Order_Time"].dt.hour

print("\nOrders by Hour:")
print(df["Order_Hour"].value_counts().sort_index())

# TASK 117: Delivery time by order status
print("\nDelivery Time by Status:")
print(df.groupby("Order_Status")["Delivery_Time"].mean())

# TASK 118: Cancellation and delay rate by city
df["Is_Cancelled"] = df["Order_Status"].str.lower() == "cancelled"
df["Is_Delayed"] = (
    (df["Delivery_Time"] > 45) |
    (df["Order_Status"].str.lower() == "delayed")
)

print("\nCancellation Rate by City:")
print(df.groupby("City")["Is_Cancelled"].mean() * 100)

print("\nDelay Rate by City:")
print(df.groupby("City")["Is_Delayed"].mean() * 100)

# TASK 119: NumPy median and sample standard deviation
times = df["Delivery_Time"].dropna()

print("\nMedian Delivery Time:", np.median(times))
print("Sample Standard Deviation:", np.std(times, ddof=1))

# TASK 120: Detect outliers using IQR
q1 = times.quantile(0.25)
q3 = times.quantile(0.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr

outliers = times[times > upper]
print("\nUpper Outlier Limit:", upper)
print("Outliers:")
print(outliers)

# TASK 121: Create delivery time categories
def delivery_category(time):
    if pd.isna(time):
        return "Not Delivered"
    elif time <= 30:
        return "Fast"
    elif time <= 45:
        return "Standard"
    else:
        return "Delayed"

df["Delivery_Category"] = df["Delivery_Time"].apply(delivery_category)

print("\nDelivery Categories:")
print(df["Delivery_Category"].value_counts())

# TASK 122: Revenue by city and payment mode
revenue_df = df[~cancelled.reindex(df.index, fill_value=False)]
revenue_df = revenue_df[
    revenue_df["Order_Status"].str.lower() != "cancelled"
]

print("\nRevenue by City and Payment Mode:")
print(revenue_df.pivot_table(
    index="City",
    columns="Payment_Mode",
    values="Order_Value",
    aggfunc="sum",
    fill_value=0
))

# TASK 123: Operational recommendations
print("\nOPERATIONAL RECOMMENDATIONS")
print("1. Improve delivery speed in cities with high delay rates.")
print("2. Increase delivery staff during peak ordering hours.")
print("3. Investigate restaurants with high cancellation rates.")
print("4. Monitor orders taking more than 45 minutes.")
print("5. Improve operations in cities with low delivery performance.")

# Save cleaned dataset
df.to_csv("cleaned_food_delivery.csv", index=False)

print("\nProject completed successfully!")