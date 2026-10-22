# 1. Create a Python file named unsupervised_ml.py
# 2. Import KMeans from sklearn.cluster and StandardScaler from sklearn.preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # <-- add this exact line right here!

# 3. Prepare the customer data using a Python list (Spending, Visits)
customer_data = [[1000,2],
 [1500, 3],
 [2000, 2],
 [5000, 6],
 [6000, 7],
 [5500, 5],
 [12000, 12],
 [15000, 14],
 [13000, 11],
 [3000, 4]
]
# 4. Use StandardScaler to scale the spending and visits features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# 5 & 6. Create a K-Means model using n_clusters=3, random_state=42, and n_init=10
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# 7. Train the model using the customer dataset
cluster_assignments = kmeans.fit_predict(scaled_data)

# 8. Display the cluster assigned to each customer matching the Example Output
for index, cluster_id in enumerate(cluster_assignments):
    print(f"Customer {index + 1} belongs to Cluster {cluster_id}")

print()  # Adds a blank line for clean formatting

# 9. Display the total number of customers in each cluster
for cluster_id in sorted(set(cluster_assignments)):
    total_customers = list(cluster_assignments).count(cluster_id)
    print(f"Cluster {cluster_id}: {total_customers} customer(s)")
# ==================== BONUS TASK ====================
# Extract spending (X) and visits (Y) for plotting
spending = [row[0] for row in customer_data]
visits = [row[1] for row in customer_data]

plt.figure(figsize=(8, 5))

# Create a scatter plot with colors corresponding to cluster IDs
scatter = plt.scatter(spending, visits, c=cluster_assignments, cmap='viridis', s=100, edgecolors='black')

# Add labels and formatting
plt.title("Customer Segmentation Groups (K-Means)")
plt.xlabel("Monthly Spending (KES)")
plt.ylabel("Monthly Visits")
plt.grid(True, linestyle='--', alpha=0.6)

# Add a legend showing the clusters
plt.legend(*scatter.legend_elements(), title="Clusters")

# Display the plot window
plt.show()