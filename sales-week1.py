import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("train.csv")

#inscept the data
head=df.head()
print("first  five item from the dataset:\n",head)
tail=df.tail()
print("last five item from the dataset:\n",tail)
des=df.describe()
print("description of the dataset:\n",des)

#convert dates
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
print(df.head)

# Check for missing values in each column
missing_values = df.isnull().sum()

# Print the count of missing values for each column
print("Missing values in each column after dropna():\n", missing_values)

# Drop rows with missing values
df = df.dropna()

# Print the DataFrame after dropping rows with missing values
print("DataFrame after dropping rows with missing values:")
print(df)

# Check if there are still any missing values
total_missing_after_drop = df.isnull().sum().sum()
print("\nTotal missing values after dropna():", total_missing_after_drop)

#total sales
total_sales = round(df['Sales'].sum(),2)
print(f"Total Sales: {total_sales}")

#Average sales
avg_sales=round(df['Sales'].mean(),2)
print(f"Average sale:{avg_sales}")

#sales trend over time
sales_trends = df.groupby('Order Date')['Sales'].sum()
print(sales_trends.head())

#line plot for sale trend over time
plt.figure(figsize=(12, 6))
sales_trends.plot(kind='line', title="Sales Trends Over Time")
plt.ylabel('Total Sales')
plt.xlabel('Order Date')
plt.show()

#Bar chart for top category/sub-category
sales_by_category = df.groupby('Category')['Sales'].sum()
sales_by_category.plot(kind='bar', figsize=(8, 5), title="Sales by Category")
plt.ylabel('Total Sales')
plt.show()

#Top products
top_products = df.groupby('Product Name')['Sales'].sum().nlargest(10)
top_products.plot(kind='bar', figsize=(10, 6), title="Top 10 Products by Sales")
plt.ylabel('Total Sales')
plt.show()

#Top region
sales_by_subcategory = df.groupby('Sub-Category')['Sales'].sum().nlargest(10)
print(sales_by_subcategory)


'''List all unique subcategories
unique_subcategories = df['Sub-Category'].nunique()

#Print the list of unique subcategories
print(f"Unique subcategories: {unique_subcategories}")'''
