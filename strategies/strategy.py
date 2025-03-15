import ast
import csv
import random
import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

GENERATION = 672

POPULATION_SIZE = 672

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('energy_optimization.log'),
        logging.StreamHandler()])
logger = logging.getLogger(__name__)


class EnergyStrategyOptimizer:
    """NSGA-II based energy strategy optimizer with mathematical formulation"""

    def __init__(self, site_info, demand_data, energy_data):
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        logger.info("Initialization ...")

        # Parse site parameters
        self._parse_site_parameters(site_info)
        self._process_time_data(demand_data, energy_data)
        self._initialize_energy_parameters(demand_data, energy_data)

        self.site_name = None
        self._toolbox()

    def _parse_site_parameters(self, site_info):
        """Extract and validate site parameters"""
        params = site_info.iloc[0]
        self.SOC_init = params['init SOC'] / 100
        self.DOD = params['DOD'] / 100
        self.SOC_min = 1 - self.DOD
        self.battery_capacity = params['battery capacity (Ah)']
        self.rated_voltage = params['rated voltage（V）']
        self.grid_power = params['grid power(kW)']
        self.diesel_power = params['diesel power(kW)']
        self.rate_charge = params['battery charge coefficient']
        self.rate_discharge = params['battery discharge coefficient']

        # Parse grid outage plan
        outage_plan = ast.literal_eval(
            params['grid outage plan']
            .replace('false', 'False')
            .replace('true', 'True')
            .replace(' ', ', ')
        )
        self.grid_available = np.array(
            [not outage_plan[h // 4] for h in range(7 * 24 * 4)])

        logger.info("Extraction and validation site parameters ...")

    def _process_time_data(self, demand_data, energy_data):
        """Process temporal data and alignment"""
        self.datetime = pd.to_datetime(demand_data["index"])
        demand_data.set_index("index", inplace=True)
        energy_data.set_index("index", inplace=True)
        self.solar_available = (
            self.datetime.dt.hour >= 6) & (
            self.datetime.dt.hour < 18)

        logger.info("Processing temporal data and alignment ...")

    def _initialize_energy_parameters(self, demand_data, energy_data):
        """Initialize energy parameters"""
        self.T = len(demand_data)
        self.Δt = 0.25  # 15-minute intervals in hours
        self.solar_energy = energy_data['Energy Output(kWh)'].values
        self.consumption = demand_data['Total Energy(kWh)'].values
        self.battery_kwh = (self.battery_capacity * self.rated_voltage) / 1000

        # Weights from challenge guidelines
        self.weights = {
            'diesel_starts': 300,
            'diesel_duration': 1,
            'diesel_max_streak': 0.95,
            'grid_duration': 0.25
        }

        logger.info("Initialization: energy parameters ...")

    def _toolbox(self):
        """Initialize toolbox"""
        # Create DEAP framework components
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register(
            "individual",
            self.init_individual)
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual)
        toolbox.register("evaluate", self.evaluate_strategy)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate_individual)
        toolbox.register("select", tools.selNSGA2)

        self.toolbox = toolbox

        logger.info("Initialization: toolbox ...")

    def update_soc(self, current_soc, p_solar, p_grid, p_diesel, p_load):
        """Update SOC according to mathematical formulation"""
        net_energy = p_solar + p_grid + p_diesel - p_load

        if net_energy >= 0:  # Charging
            ΔSOC = (net_energy * self.Δt * self.rate_charge) / self.battery_kwh
        else:  # Discharging
            ΔSOC = (net_energy * self.Δt) / \
                (self.battery_kwh * self.rate_discharge)

        new_soc = current_soc + ΔSOC
        new_soc = np.clip(new_soc, self.SOC_min, 1.0)
        return new_soc

    def evaluate_strategy(self, individual):
        """Multi-objective evaluation function"""
        metrics = {
            'diesel_starts': 0,
            'diesel_duration': 0,
            'diesel_max_streak': 0,
            'grid_duration': 0,
            'soc_violations': 0,
            'current_streak': 0
        }

        soc = self.SOC_init
        diesel_running = False

        for t in range(self.T):
            # Decode energy source decisions
            decision = individual[t]
            solar = (decision & 1) and self.solar_available[t]
            grid = (decision & 2) and self.grid_available[t]
            diesel = (decision & 4) != 0

            # Calculate power values
            p_solar = self.solar_energy[t] if solar else 0
            p_grid = self.grid_power * self.Δt if grid else 0
            p_diesel = self.diesel_power * self.Δt if diesel else 0
            p_load = self.consumption[t]

            # Update SOC and track violations
            soc = self.update_soc(soc, p_solar, p_grid, p_diesel, p_load)
            if soc <= self.SOC_min:
                metrics['soc_violations'] += 1

            # Track diesel metrics
            if diesel:
                metrics['diesel_duration'] += 1
                metrics['current_streak'] += 1
                if not diesel_running:
                    metrics['diesel_starts'] += 1
                    diesel_running = True
                metrics['diesel_max_streak'] = max(
                    metrics['diesel_max_streak'], metrics['current_streak'])
            else:
                metrics['current_streak'] = 0
                diesel_running = False

            # Track grid usage
            metrics['grid_duration'] += 1 if grid else 0

        # Calculate objectives
        cost = (
            self.weights['diesel_starts'] * metrics['diesel_starts'] +
            self.weights['diesel_duration'] * metrics['diesel_duration'] +
            self.weights['diesel_max_streak'] * metrics['diesel_max_streak'] +
            self.weights['grid_duration'] * metrics['grid_duration']
        )
        penalty = metrics['soc_violations']

        return cost, penalty

    def optimize(
            self,
            generations=GENERATION,
            pop_size=POPULATION_SIZE,
            site_name='Site1'):
        """NSGA-II based optimization with constraint handling"""
        logger.info(
            "%s: NSGA-II based optimization with constraint handling ...",
            site_name.upper())
        self.site_name = site_name

        # Initialize population
        population, _ = self.load_checkpoint(
            site_name, pop_size)  # start_gen = _

        # Setup optimization tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("avg", np.mean, axis=0)
        stats.register("median", np.median, axis=0)
        stats.register("max", np.max, axis=0)
        pareto_front = tools.ParetoFront()

        # Run NSGA-II optimization
        population, _ = algorithms.eaMuPlusLambda(  # logbook = _
            population, self.toolbox,
            mu=pop_size,
            lambda_=pop_size,
            cxpb=0.7,
            mutpb=0.2,
            ngen=generations,
            stats=stats,
            halloffame=pareto_front,
            verbose=True
        )

        # Save final state
        self.save_checkpoint(site_name, population, generations)

        # Select best feasible solution
        feasible = [
            ind for ind in pareto_front if ind.fitness.values[1] < 1e-6]
        if feasible:
            best = min(feasible, key=lambda x: x.fitness.values[0])
            logger.info(
                "Found feasible solution with cost: %.2f",
                best.fitness.values[0])
        else:
            best = min(pareto_front, key=lambda x: x.fitness.values[0])
            logger.warning("No feasible solutions! Using best infeasible")

        return best

    def init_individual(self, _):
        """Initialize individual respecting constraints"""
        individual = []
        for t in range(self.T):
            solar_bit = random.randint(0, 1) if self.solar_available[t] else 0
            grid_bit = random.randint(0, 1) if self.grid_available[t] else 0
            diesel_bit = random.randint(0, 1)
            individual.append(solar_bit | (grid_bit << 1) | (diesel_bit << 2))

        return creator.Individual(individual)

    def mutate_individual(self, individual):
        """Constrained mutation operator"""
        for t in random.sample(range(self.T), k=int(self.T * 0.1)):
            # Solar mutation
            if self.solar_available[t]:
                individual[t] ^= 1
            # Grid mutation
            if self.grid_available[t]:
                individual[t] ^= 2
            # Diesel mutation
            individual[t] ^= 4

        return individual,

    def save_strategy(self, individual, filename):
        """Save optimized strategy to CSV"""
        with open(filename, 'w', newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["site name", "time", "solar", "grid", "diesel"])
            for t, decision in enumerate(individual):
                solar = "TRUE" if (
                    decision & 1) and self.solar_available[t] else "FALSE"
                grid = "TRUE" if (
                    decision & 2) and self.grid_available[t] else "FALSE"
                diesel = "TRUE" if (decision & 4) else "FALSE"
                writer.writerow([self.site_name, t, solar, grid, diesel])

            logger.info("Optimized strategy saved to %s", filename)

    def load_checkpoint(self, site_name, pop_size):
        """Load population from checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{site_name}_checkpoint.pkl"
        if checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
                logger.info(
                    "Loaded checkpoint from generation %d", data['generation'])
                return data["population"], data["generation"] + 1
            logger.warning("Checkpoint not found, starting fresh")
        return self.toolbox.population(n=pop_size), 0

    def save_checkpoint(self, site_name, population, generation):
        """Save current optimization state"""
        checkpoint_path = self.checkpoint_dir / f"{site_name}_checkpoint.pkl"
        data = {
            "generation": generation,
            "population": population,
            "parameters": {
                "SOC_init": self.SOC_init,
                "DOD": self.DOD,
                "battery_kwh": self.battery_kwh
            }
        }
        with open(checkpoint_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved checkpoint at generation %d", generation)

# Usage example:
# strategy = EnergyStrategyOptimizer(site_info, demand_df, energy_df)
# best_strategy = strategy.optimize(generations=100, site_name="Site1")
# strategy.save_strategy(best_strategy, "optimal_strategy.csv")
