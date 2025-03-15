import pandas as pd
# from strategy_v2 import EnergyStrategyOptimized
from strategy import EnergyStrategyOptimizer

for i in range(1, 11):
    # Load the datasets
    site_info = pd.read_csv(f'data/input/site{i}/site_grid_outage.csv')
    demand_df = pd.read_csv(f'data/output/site{i}/pred_demand.csv')
    energy_df = pd.read_csv(f"data/output/site{i}/pred_solar.csv")

    # Initialize and run the strategy using init SOC and DOD from the dataset
    strategy = EnergyStrategyOptimizer(site_info, demand_df, energy_df)
    best_strategy = strategy.optimize(generations=100, site_name=f'site{i}')
    strategy.save_strategy(best_strategy, f'strategy{i}.csv')

df_combined = pd.read_csv("strategy1.csv", dtype={
        "site name": "object",  # Keep as string
        "time": "int64",        # Keep as integer
        "solar": "object",      # Force as string
        "grid": "object",       # Force as string
        "diesel": "object"      # Force as string
    })
for i in range(2, 11):
    df_next = pd.read_csv(f"strategy{i}.csv", dtype={
        "site name": "object",  # Keep as string
        "time": "int64",        # Keep as integer
        "solar": "object",      # Force as string
        "grid": "object",       # Force as string
        "diesel": "object"      # Force as string
    })
    df_combined = pd.concat([df_combined, df_next], axis=0)

df_combined.to_csv("strategy.csv", index=False)

print("end ...")
