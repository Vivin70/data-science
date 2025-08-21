import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
bookings = pd.read_csv(r"C:\Users\vivin\OneDrive\Desktop\project 1\Bookings.csv")
sessions = pd.read_csv(r"C:\Users\vivin\OneDrive\Desktop\project 1\Sessions.csv")

# 1. Distinct counts
distinct_bookings = bookings['booking_id'].nunique()
distinct_sessions = sessions['session_id'].nunique()
distinct_searches = sessions['search_id'].nunique()
print(f"Distinct bookings: {distinct_bookings}")
print(f"Distinct sessions: {distinct_sessions}")
print(f"Distinct searches: {distinct_searches}")

# 2. Sessions with more than one booking
multi_booking_sessions = sessions.groupby('session_id').filter(lambda x: x['booking_id'].nunique() > 1).shape[0]
print(f"Sessions with more than one booking: {multi_booking_sessions}")

# 3. Day of the week analysis
bookings['booking_time'] = pd.to_datetime(bookings['booking_time'])
bookings['day_of_week'] = bookings['booking_time'].dt.day_name()
day_counts = bookings['day_of_week'].value_counts()
print("Bookings by day of the week:")
print(day_counts)

day_counts.plot.pie(autopct='%1.1f%%', startangle=90, figsize=(8, 8))
plt.title("Bookings by Day of the Week")
plt.ylabel('')
plt.show()

# 4. Bookings and Gross Booking Value by service
service_data = bookings.groupby('service_name').agg(
    total_bookings=('booking_id', 'count'),
    total_gross_booking_value=('INR_Amount', 'sum')
)
print("Service-wise bookings and gross booking value:")
print(service_data)

print('_______________________________________')

# 5. Most booked route
customer_bookings = bookings.groupby('customer_id').filter(lambda x: len(x) > 1)
most_booked_route = customer_bookings.groupby(['from_city', 'to_city']).size().idxmax()
print(f"Most booked route for customers with more than 1 booking: {most_booked_route}")

# 6. Top 3 departure cities for advance bookings
advance_cities = bookings[bookings['days_to_departure'] > 0].groupby('from_city').size()
top_3_cities = advance_cities[advance_cities >= 5].nlargest(3)
print("Top 3 departure cities for advance bookings:")
print(top_3_cities)

# 7. Heatmap for correlations
numerical_cols = ['no_of_passengers', 'INR_Amount', 'days_to_departure', 'distance_km']
correlation = bookings[numerical_cols].corr()
print("Correlation matrix:")
print(correlation)

sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title("Correlation Heatmap")
plt.show()

# 8. Device type by service
device_usage = bookings.groupby(['service_name', 'device_type_used']).size()
most_used_devices = device_usage.groupby('service_name').idxmax()
print("Most used device type for each service:")
print(most_used_devices)

# 9. Quarterly trends
bookings['quarter'] = bookings['booking_time'].dt.to_period('Q')
quarterly_trends = bookings.groupby(['quarter', 'device_type_used']).size().unstack()
print("Quarterly trends for bookings by device type:")
print(quarterly_trends)

quarterly_trends.plot(kind='line', marker='o', figsize=(10, 6))
plt.title("Quarterly Booking Trends by Device Type")
plt.xlabel("Quarter")
plt.ylabel("Number of Bookings")
plt.legend(title="Device Type")
plt.grid(True)
plt.show()

# 10. oBSR Analysis
bookings_daily = bookings.groupby(bookings['booking_time'].dt.date).size()
sessions_daily = sessions.groupby(sessions['search_time'].str[:10]).size()

oBSR = bookings_daily / sessions_daily.reindex(bookings_daily.index, fill_value=0)
print("Daily oBSR:")
print(oBSR)

monthly_oBSR = oBSR.groupby(pd.to_datetime(oBSR.index).month).mean()
print("Average monthly oBSR:")
print(monthly_oBSR)

oBSR.plot(figsize=(10, 6), marker='o')
plt.title("Daily oBSR Trends")
plt.xlabel("Date")
plt.ylabel("oBSR")
plt.grid(True)
plt.show()
