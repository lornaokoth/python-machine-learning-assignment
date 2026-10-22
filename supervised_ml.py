import numpy as np
from sklearn.linear_model import LinearRegression

hours_studied = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
exam_scores = [35, 40, 48, 52, 60, 68, 75, 81, 89, 95]

X = [[hour] for hour in hours_studied]
y = exam_scores

model = LinearRegression()
model.fit(X, y)

print("--- Student Exam Score Predictor Model Trained ---")
try:
    user_hours = float(input("Enter the number of hours studied: "))
    prediction = model.predict([[user_hours]])
    print(f"Predicted Exam Score: {prediction[0]:.2f}")
except ValueError:
    print("Please enter a valid numeric value for hours.")

import matplotlib.pyplot as plt
import numpy as np

# 1. Plot the actual data points from your dataset
# Replace 'X' with your study hours variable and 'y' with your actual exam scores variable
plt.scatter(X, y, color='blue', label='Actual Data')
X_array = np.array(X) 

# 2. Plot the regression line (prediction line)
# Replace 'model' with the name of your trained linear regression model variable
X_line = np.linspace(X_array.min(), X_array.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')

# 3. Add titles, labels, and formatting
plt.title('Relationship Between Study Hours and Exam Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Examination Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 4. Display the plot
plt.show()