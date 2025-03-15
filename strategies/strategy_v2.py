import ast
import csv
import random
import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('energy_strategy_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnergyStrategyOptimized:
    """Optimized Energy strategy optimization class"""
    def __init__(self, _site_info, _demand, _energy_data):
        self.checkpoint_dir = Path("optimized_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        logger.info("Initialized EnergyStrategyOptimized")

        self.datetime = pd.to_datetime(_demand["index"])
        _demand.set_index("index", inplace=True)
        _energy_data.set_index("index", inplace=True)

        # Site parameters
        self.SOC_t = _site_info['init SOC'].values[0] / 100
        self.DOD = _site_info['DOD'].values[0] / 100
        self.rated_capacity = _site_info['battery capacity (Ah)'].values[0]
        self.rated_voltage = _site_info['rated voltage（V）'].values[0]
        self.grid_power = _site_info['grid power(kW)'].values[0]
        self.diesel_power = _site_info['diesel power(kW)'].values[0]
        self.charge_rate = _site_info['battery charge coefficient'].values[0]
        self.discharge_rate = _site_info['battery discharge coefficient'].values[0]

        # Grid outage plan
        outage_plan = ast.literal_eval(
            _site_info['grid outage plan'].values[0]
            .replace('false', 'False')
            .replace('true', 'True')
            .replace(' ', ', ')
        )
        self.grid_available = np.array([not outage_plan[t // 4] for t in range(7 * 24 * 4)])

        # Load data
        self.solar = _energy_data['Energy Output(kWh)'].values
        self.consumption = _demand['Total Energy(kWh)'].values
        self.T = len(self.solar)

        # Battery parameters
        self.battery_factor = (
            self.rated_capacity * self.rated_voltage) / 1000 * 0.25  # kWh per 15min at SOC=1
        self.loss_penalty_factor = 1000  # Penalty per kWh lost

    def update_soc(self, current_soc, net_energy):
        """Correctly update SOC based on net energy and battery parameters"""
        if net_energy >= 0:
            delta_soc = (
                net_energy * self.charge_rate * 1000) / (
                    self.rated_capacity * self.rated_voltage)
        else:
            delta_soc = (
                net_energy / self.discharge_rate * 1000) / (
                    self.rated_capacity * self.rated_voltage)
        new_soc = current_soc + delta_soc
        lost_energy = max(0, new_soc - 1) * self.battery_factor
        new_soc = np.clip(new_soc, 0, 1)
        return new_soc, lost_energy

    def evaluate(self, individual):
        """Evaluate with corrected SOC penalty calculation"""
        soc = self.SOC_t
        total_penalty = 0.0
        diesel_runs = 0
        diesel_duration = 0
        grid_usage = 0
        current_diesel_streak = 0
        max_diesel_streak = 0
        total_energy_loss = 0.0
        diesel_running_prev = False
        min_soc = 1 - self.DOD

        for t in range(self.T):
            decision = individual[t]
            hour = self.datetime[t].hour
            solar_available = 6 <= hour < 18
            grid_available = self.grid_available[t]

            solar_used = (decision & 1) != 0 and solar_available
            grid_used = (decision & 2) != 0 and grid_available
            diesel_used = (decision & 4) != 0

            solar_supply = self.solar[t] if solar_used else 0.0
            grid_supply = self.grid_power * 0.25 if grid_used else 0.0
            diesel_supply = self.diesel_power * 0.25 if diesel_used else 0.0

            total_supply = solar_supply + grid_supply + diesel_supply
            load = self.consumption[t]
            net_energy = total_supply - load

            new_soc, lost_energy = self.update_soc(soc, net_energy)
            total_energy_loss += lost_energy * self.loss_penalty_factor

            # Corrected penalty calculation
            if new_soc < min_soc:
                penalty = (min_soc - new_soc) * 1e6
                total_penalty += penalty

            # Track diesel usage
            if diesel_used:
                diesel_duration += 1
                current_diesel_streak += 1
                if not diesel_running_prev:
                    diesel_runs += 1
                diesel_running_prev = True
                max_diesel_streak = max(max_diesel_streak, current_diesel_streak)
            else:
                current_diesel_streak = 0
                diesel_running_prev = False

            if grid_used:
                grid_usage += 1

            soc = new_soc

        cost = (300 * diesel_runs +
                1 * diesel_duration +
                0.95 * max_diesel_streak +
                0.25 * grid_usage)

        logger.debug("Evaluation results - Cost: %.2f, Penalty: %.2f", cost, total_penalty)
        return (cost, total_penalty + total_energy_loss)

    def optimize_strategy(self, generations=100, pop_size=300, site_name='Site1'):
        """Optimize energy strategy using NSGA-II algorithm with proper constraint handling"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # Custom individual initialization
        def init_individual():
            individual = []
            for t in range(self.T):
                hour = self.datetime[t].hour
                solar_avail = 6 <= hour < 18
                grid_avail = self.grid_available[t]
                solar_bit = random.randint(0, 1) if solar_avail else 0
                grid_bit = random.randint(0, 1) if grid_avail else 0
                diesel_bit = random.randint(0, 1)
                decision = solar_bit | (grid_bit << 1) | (diesel_bit << 2)
                individual.append(decision)
            return individual

        # Register genetic operators
        toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate, mutpb=0.15)
        toolbox.register("select", tools.selNSGA2)

        # Setup checkpoint directory
        checkpoint_path = self.checkpoint_dir / f"{site_name}_checkpoint.pkl"
        population = None

        # Try to load checkpoint
        if checkpoint_path.exists():
            with open(checkpoint_path, "rb") as cp_file:
                checkpoint = pickle.load(cp_file)
                population = checkpoint["population"]
                start_gen = checkpoint["generation"] + 1
                logger.info("Resuming from checkpoint generation %d", start_gen-1)
        else:
            start_gen = 0
            population = toolbox.population(n=pop_size)

        # Initialize statistics and Pareto front
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("avg", np.mean, axis=0)
        stats.register("max", np.max, axis=0)
        pareto_front = tools.ParetoFront()

        # Run NSGA-II optimization
        population, _ = algorithms.eaMuPlusLambda( # logbook = _
            population, toolbox,
            mu=pop_size,  # Number of parents to keep
            lambda_=pop_size,  # Number of offspring to produce
            cxpb=0.7,  # Crossover probability
            mutpb=0.2,  # Mutation probability
            ngen=generations,  # Total generations to run
            stats=stats,
            halloffame=pareto_front,
            verbose=True
        )

        # Save final checkpoint
        checkpoint = {
            "generation": generations,
            "population": population,
            "pareto_front": pareto_front
        }
        with open(checkpoint_path, "wb") as cp_file:
            pickle.dump(checkpoint, cp_file)

        # Select best feasible solution
        feasible = [ind for ind in pareto_front if ind.fitness.values[1] <= 1e-6]
        if feasible:
            best_ind = min(feasible, key=lambda x: x.fitness.values[0])
            logger.info("Found feasible solution with cost: %.2f", best_ind.fitness.values[0])
        else:
            best_ind = min(pareto_front, key=lambda x: x.fitness.values[0])
            logger.warning("No fully feasible solutions found! Selected best infeasible solution")

        return best_ind

    def mutate(self, individual, mutpb):
        """Mutation respecting availability constraints"""
        t = 0
        while t < len(individual):
            if random.random() < mutpb:
                hour = self.datetime[t].hour
                solar_avail = 6 <= hour < 18
                grid_avail = self.grid_available[t]
                # Flip solar bit if available
                if solar_avail:
                    individual[t] ^= 1
                # Flip grid bit if available
                if grid_avail:
                    individual[t] ^= 2
                # Flip diesel bit
                individual[t] ^= 4

            t += 1
        return individual,

    def run_strategy(self, filename="optimized_strategy.csv", site_name='Site1'):
        """Execute optimization and save results"""
        logger.info("Starting optimized strategy for %s", site_name)
        try:
            best_strategy = self.optimize_strategy(site_name=site_name)
            logger.info("Optimization completed")
        except Exception as e:
            logger.error("Optimization failed: %s", str(e))
            raise

        with open(filename, 'w', newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["site name", "time", "solar", "grid", "diesel"])
            for t, decision in enumerate(best_strategy):
                hour = self.datetime[t].hour
                solar_avail = 6 <= hour < 18
                solar_used = (decision & 1) != 0 and solar_avail
                grid_used = (decision & 2) != 0 and self.grid_available[t]
                diesel_used = (decision & 4) != 0
                writer.writerow([
                    site_name,
                    t,
                    "TRUE" if solar_used else "FALSE",
                    "TRUE" if grid_used else "FALSE",
                    "TRUE" if diesel_used else "FALSE"
                ])
        logger.info("Optimized strategy saved to %s", filename)

# Example usage:
# Assuming site_info, demand, energy_data are loaded DataFrames
# strategy = EnergyStrategyOptimized(site_info, demand, energy_data)
# strategy.run_strategy()
