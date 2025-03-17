import csv
import random

# Define constants
NUM_ROWS = 10000  # Number of rows to generate
WEAPON_TYPES = ["Shotgun", "Missile", "Cannon", "Drone", "Artillery", "Interceptor"]
TARGET_TYPES = ["Vehicle", "Aircraft", "Infantry", "Bunker", "Missile"]
PRIORITIES = ["High", "Medium", "Low"]

# Define Adjustment Factors for Assigned Time based on Weapon Type and Priority
ADJUSTMENT_FACTORS = {
    "Shotgun": {"High": 10, "Medium": 20, "Low": 30},
    "Missile": {"High": 60, "Medium": 120, "Low": 180},
    "Cannon": {"High": 30, "Medium": 60, "Low": 90},
    "Drone": {"High": 20, "Medium": 40, "Low": 60},
    "Artillery": {"High": 40, "Medium": 80, "Low": 120},
    "Interceptor": {"High": 50, "Medium": 100, "Low": 150},
}

# Function to generate random data
def generate_random_data(num_rows):
    data = []
    for i in range(num_rows):
        weapon_id = f"W{i+1:05}"  # Weapon ID with leading zeros
        weapon_type = random.choice(WEAPON_TYPES)
        weapon_speed = random.randint(200, 2000)  # Random speed between 200 and 2000 m/s
        target_distance = random.randint(1000, 50000)  # Random target distance between 1000 and 50000 meters
        target_type = random.choice(TARGET_TYPES)
        priority = random.choice(PRIORITIES)
        adjustment_factor = ADJUSTMENT_FACTORS[weapon_type][priority]  # Get adjustment factor
        assigned_time = (target_distance / weapon_speed) + adjustment_factor  # Calculate Assigned Time in seconds
        data.append([weapon_id, weapon_type, weapon_speed, target_distance, target_type, priority, round(assigned_time, 2)])
    return data

# Generate data
data = generate_random_data(NUM_ROWS)

# Write data to CSV file
filename = "weapon_target_data.csv"
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Weapon ID", "Weapon Type", "Weapon Speed (m/s)", "Target Distance (m)", "Target Type", "Priority", "Assigned Time (s)"])
    writer.writerows(data)

print(f"Generated {NUM_ROWS} rows of data in '{filename}'")