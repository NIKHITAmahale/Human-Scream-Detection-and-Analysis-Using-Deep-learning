import pandas as pd
import matplotlib.pyplot as plt

# Load the log file
log_file = 'scream_detection_log.csv'
data = pd.read_csv(log_file, parse_dates=['Timestamp'])

# Plot probability over time
plt.figure(figsize=(10, 5))
plt.plot(data['Timestamp'], data['Probability'].astype(float), marker='o', linestyle='-')
plt.title('Scream Detection Probabilities Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Probability')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
