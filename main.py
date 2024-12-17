import gym_cutting_stock
import gymnasium as gym
import numpy as np
from src.policy.policyCall import PolicyCall, arr_policy
import matplotlib.pyplot as plt
import sys
import time

plt.ion()

def print_usage():
    print("Usage: python main.py [index]")
    print("\nAvailable policies:")
    for idx, policy in enumerate(arr_policy):
        print(f"{idx}: {policy.__name__}")

def main(index_of_policy=0):
    if index_of_policy >= len(arr_policy):
        print("Please provide a valid index.")
        print_usage()
        sys.exit(1)

    print(f"Using policy: {arr_policy[index_of_policy].__name__}")
    policy = PolicyCall(index_of_policy)

    # Create environment with smaller dimensions for testing
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        render_mode="human",
        min_w=20,
        min_h=20,
        max_w=50,
        max_h=50,
        num_stocks=100,
    )
    
    # Track metrics
    total_products = 0
    placed_products = 0
    used_stocks = set()
    start_time = time.time()

    observation, info = env.reset(seed=42)
    stocks = observation["stocks"]

    while True:
        # Count remaining products
        products = observation["products"]
        remaining = sum(p["quantity"] for p in products)
        
        if remaining == 0:
            break
            
        if total_products == 0:
            total_products = remaining

        # Get action from policy
        action = policy.get_action(observation, info)
        
        if action["stock_idx"] == -1:
            print("No valid placement found")
            break

        # Track used stocks
        used_stocks.add(action["stock_idx"])
        
        # Apply action
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Update metrics
        if reward > 0:
            placed_products += 1
            
        print(f"Action: {action}")
        print(f"Products remaining: {remaining}")
        print(f"Stocks used: {len(used_stocks)}")
        print(f"Filled ratio: {info['filled_ratio']:.2%}")
        print(f"Trim loss: {info['trim_loss']:.2%}")
        print("-" * 50)

        if terminated or truncated:
            break

    # Print final results
    elapsed_time = time.time() - start_time
    print("\nFinal Results:")
    print(f"Total products: {total_products}")
    print(f"Placed products: {placed_products}")
    print(f"Used stocks: {len(used_stocks)}")
    print(f"Time taken: {elapsed_time:.2f}s")
    print(f"Final fill ratio: {info['filled_ratio']:.2%}")
    print(f"Final trim loss: {info['trim_loss']:.2%}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ("-h", "--help"):
            print_usage()
            sys.exit(0)
        try:
            index = int(sys.argv[1])
        except ValueError:
            print("Please provide a valid integer index.")
            print_usage()
            sys.exit(1)
        main(index_of_policy=index)
    else:
        main()