import numpy as np
import random
import pickle5 as pickle

def generate_fire_spread(grid_size=(50, 50), timesteps=50, init_fire_points=5,
                           base_spread_prob=0.3, wind_speed_range=(0, 10),
                           wind_direction_range=(0, 360), humidity_range=(10, 90),
                           jump_threshold=7, wind_variation=2, humidity_variation=5):
    """
    Generates a synthetic dataset simulating fire spread over time with probabilistic BFS.
    
    Parameters:
        grid_size: Tuple (height, width) of the fire spread grid.
        timesteps: Number of timesteps to simulate.
        init_fire_points: Number of initial fire pixels.
        base_spread_prob: Base probability of fire spreading to a neighbor.
        wind_speed_range: Min and max wind speed (affects spread distance).
        wind_direction_range: Min and max wind direction (0 to 360 degrees).
        humidity_range: Min and max humidity (higher humidity reduces spread probability).
        jump_threshold: Wind speed above which fire can jump cells instead of normal BFS.
        wind_variation: Maximum change in wind speed per timestep.
        humidity_variation: Maximum change in humidity per timestep.
    
    Returns:
        dataset: List of (fire_grid, weather_conditions) tuples for each timestep.
    """
    height, width = grid_size
    fire_grid = np.zeros((height, width), dtype=int)  # 0 = no fire, 255 = fire
    dataset = []
    
    # Initialize fire in random positions
    fire_positions = random.sample([(i, j) for i in range(height) for j in range(width)], init_fire_points)
    for i, j in fire_positions:
        fire_grid[i, j] = 255
    
    def get_neighbors(i, j):
        """Returns valid neighbor coordinates."""
        neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1),
                     (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
        return [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]
    
    # Initialize weather conditions
    wind_speed = random.uniform(*wind_speed_range)
    wind_direction = random.uniform(*wind_direction_range)
    humidity = random.uniform(*humidity_range)
    
    for t in range(timesteps):
        # Update weather conditions gradually
        wind_speed = max(wind_speed_range[0], min(wind_speed_range[1], wind_speed + random.uniform(-wind_variation, wind_variation)))
        wind_direction = (wind_direction + random.uniform(-wind_variation * 5, wind_variation * 5)) % 360
        humidity = max(humidity_range[0], min(humidity_range[1], humidity + random.uniform(-humidity_variation, humidity_variation)))
        
        weather_conditions = {
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "humidity": humidity
        }
        
        new_fire = []
        for i in range(height):
            for j in range(width):
                if fire_grid[i, j] == 255:  # Fire is already present
                    for ni, nj in get_neighbors(i, j):
                        if fire_grid[ni, nj] == 0:  # Not on fire yet
                            spread_prob = base_spread_prob * (1 - humidity / 100)
                            
                            # Wind effect: Fire spreads more easily in wind direction
                            angle = np.arctan2(ni - i, nj - j) * 180 / np.pi  # Compute angle
                            angle_diff = abs((angle - wind_direction) % 360)
                            if angle_diff < 45 or angle_diff > 315:
                                spread_prob *= 1.1  # More likely to spread in wind direction
                            elif angle_diff > 135 and angle_diff < 225:
                                spread_prob *= 0.5  # Less likely in opposite direction
                            
                            if random.random() < spread_prob:
                                new_fire.append((ni, nj))
        
        # Wind-based jumps if wind is strong
        if wind_speed > jump_threshold:
            for _ in range(int(wind_speed / 2)):
                ni, nj = random.randint(0, height-1), random.randint(0, width-1)
                if fire_grid[ni, nj] == 0:
                    fire_grid[ni, nj] = 255
        
        for ni, nj in new_fire:
            fire_grid[ni, nj] = 255
        
        dataset.append((fire_grid.copy(), weather_conditions))
    
    # replace 255 with 1
    fire_grid[fire_grid == 255] = 1

    return dataset



# TRAIN
n_fires = 1000

dataset = []
for i in range(n_fires):

    fire = generate_fire_spread(init_fire_points= random.randint(1, 10))
    dataset.append(fire)
    print("Generated dataset for fire", i+1)

with open("dataset/fire_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)
    

# TEST
test_dataset = []
n_fires_test = 100
for i in range(n_fires_test):

    test_fire = generate_fire_spread(init_fire_points= random.randint(1, 10))
    test_dataset.append(test_fire)
    print("Generated dataset for test fire", i+1)

with open("dataset/fire_dataset_test.pkl", "wb") as ft:
    pickle.dump(test_dataset, ft)