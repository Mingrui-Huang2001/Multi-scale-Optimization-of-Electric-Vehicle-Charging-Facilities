import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os  # Import os module to handle file paths


class HomeChargingPileModel:
    """Home Charging Pile Sharing Mechanism Model"""

    def __init__(self, lambda0=30, delta_lambda=20, p0=2.0, beta=0.2, total_piles=200, min_price_ratio=0.5):
        """
        Initialize model parameters

        Parameters:
            lambda0: Base arrival rate of user requests
            delta_lambda: Amplitude of arrival rate fluctuation over time
            p0: Base pricing for charging service
            beta: Price sensitivity coefficient (reflects user response to price changes)
            total_piles: Total number of home charging piles in the system
            min_price_ratio: Minimum price ratio relative to the base price (prevents ultra-low pricing)
        """
        self.lambda0 = lambda0
        self.delta_lambda = delta_lambda
        self.mu = 14.0  # Peak arrival time (14:00 in 24-hour format)
        self.sigma = 2.0  # Time window width for peak arrival (2 hours)

        self.p0 = p0
        self.beta = beta
        self.total_piles = total_piles
        self.min_price = min_price_ratio * p0  # Minimum charging price (avoids negative values)

        # Parameters for charging duration distribution (truncated normal distribution)
        self.charging_mu = 3.0  # Average charging duration (3 hours)
        self.charging_sigma = 1.5  # Standard deviation of charging duration (1.5 hours)
        self.charging_min = 1.0  # Minimum allowable charging duration (1 hour)
        self.charging_max = 6.0  # Maximum allowable charging duration (6 hours)

        # Parameters for owner's private usage time window
        self.private_start_base = 20.0  # Base start time of owner's private use (20:00)
        self.private_end_base = 8.0     # Base end time of owner's private use (08:00 next day)

        # Power grid constraint parameters
        self.V0 = 220.0  # Base voltage of the power grid (volts)
        self.V_min = 0.95 * self.V0  # Minimum allowable voltage (95% of base voltage)
        self.V_max = 1.05 * self.V0  # Maximum allowable voltage (105% of base voltage)

        # Initialize random buffer time (τ) for each charging pile (individual variation in private time)
        self.tau_list = np.random.uniform(0, 1, total_piles)

    def arrival_rate(self, t):
        """
        Calculate the time-varying arrival rate λ(t) of user charging requests

        Parameters:
            t: Current time in 24-hour format (e.g., 8:00 = 8.0, 18:30 = 18.5)

        Returns:
            float: Arrival rate λ(t) at time t (requests per hour)
        """
        # Ensure time is within the [0, 24) range (handles midnight crossover)
        t = t % 24

        # Non-homogeneous Poisson process (Gaussian distribution for peak shaping)
        exponent = -((t - self.mu) ** 2) / (2 * self.sigma ** 2)
        return self.lambda0 + self.delta_lambda * np.exp(exponent)

    def generate_charging_duration(self, n=1):
        """
        Generate random charging durations using a truncated normal distribution

        Parameters:
            n: Number of charging duration samples to generate

        Returns:
            np.ndarray: Array of charging durations (in hours) with length n
        """
        # Calculate truncation bounds for the normal distribution
        lower_bound = (self.charging_min - self.charging_mu) / self.charging_sigma
        upper_bound = (self.charging_max - self.charging_mu) / self.charging_sigma

        # Generate truncated normal samples
        return stats.truncnorm.rvs(
            lower_bound,
            upper_bound,
            loc=self.charging_mu,
            scale=self.charging_sigma,
            size=n
        )

    def is_owner_using(self, t, tau):
        """
        Determine if the pile owner is using the charging pile at time t

        Parameters:
            t: Current time in 24-hour format
            tau: Random buffer time for the specific charging pile

        Returns:
            bool: True if the owner is using the pile (not shareable), False otherwise
        """
        t = t % 24
        # Adjust private time window with buffer time τ
        private_start = self.private_start_base + tau
        private_end = self.private_end_base - tau

        # Handle overnight private time (e.g., 20:30 to 07:30 next day)
        if private_start < private_end:
            return private_start <= t <= private_end
        else:
            return t >= private_start or t <= private_end

    def effective_capacity(self, t, availability=None):
        """
        Calculate the effective shareable capacity N_eff(t) at time t

        Parameters:
            t: Current time in 24-hour format
            availability: Binary array (length = total_piles) indicating if each pile is functional
                          (1 = functional, 0 = malfunctioning; default: all functional)

        Returns:
            int: Number of effectively shareable charging piles at time t
        """
        # Default to all piles being functional if availability is not provided
        if availability is None:
            availability = np.ones(self.total_piles)

        # Check shareability of each pile (functional + not used by owner)
        shareable_piles = []
        for i in range(self.total_piles):
            owner_occupancy = self.is_owner_using(t, self.tau_list[i])
            shareable_piles.append(availability[i] and not owner_occupancy)

        return sum(shareable_piles)

    def dynamic_pricing(self, rho):
        """
        Calculate dynamic charging price p(t) based on real-time utilization rate

        Parameters:
            rho: Real-time utilization rate of shareable piles (arrival rate / service capacity)

        Returns:
            float: Dynamic price (ensured to be ≥ minimum price)
        """
        # Base price formula (adjusts with utilization rate, targeted at 70% optimal utilization)
        base_calculated_price = self.p0 * (1 + (rho - 0.7) / 0.3)
        # Enforce minimum price constraint
        return max(base_calculated_price, self.min_price)

    def voltage_constraint_satisfied(self, voltage):
        """
        Check if the power grid voltage meets the safe operation constraint

        Parameters:
            voltage: Actual node voltage of the power grid (volts)

        Returns:
            bool: True if voltage is within safe range, False otherwise
        """
        return self.V_min <= voltage <= self.V_max

    def current_constraint_satisfied(self, current, rated_current):
        """
        Check if the line current meets the thermal stability constraint

        Parameters:
            current: Actual line current (amps)
            rated_current: Rated current capacity of the line (amps)

        Returns:
            bool: True if current is within safe range, False otherwise
        """
        # 80% of rated current as the safety threshold
        return current <= 0.8 * rated_current

    def average_waiting_time(self, t, arrival_rate, service_rate, availability=None):
        """
        Calculate the average user waiting time W_q using queueing theory (M/M/c model variant)

        Parameters:
            t: Current time in 24-hour format
            arrival_rate: Instantaneous arrival rate λ(t) (requests per hour)
            service_rate: Service rate μ (1 / average charging duration, requests per hour per pile)
            availability: Binary array indicating functional status of each pile

        Returns:
            float: Average waiting time (hours); infinity if no available piles or utilization ≥ 1
        """
        # Get effective shareable capacity
        effective_piles = self.effective_capacity(t, availability)

        # No available piles → infinite waiting time
        if effective_piles == 0:
            return float('inf')

        # Calculate utilization rate of the effective capacity
        utilization_rate = arrival_rate / (effective_piles * service_rate)

        # Utilization rate ≥ 1 → queue overflows → infinite waiting time
        if utilization_rate >= 1:
            return float('inf')

        # Get dynamic price for waiting time adjustment (price sensitivity included)
        dynamic_price = self.dynamic_pricing(utilization_rate)

        # Formula for average waiting time (adjusted for price sensitivity)
        numerator = utilization_rate * np.sqrt(2 * (effective_piles + 1))
        denominator = service_rate * effective_piles * (1 - utilization_rate) * (1 - utilization_rate ** effective_piles) * (1 + self.beta * dynamic_price)

        return numerator / denominator

    def simulate_day(self, time_steps=1000):
        """
        Simulate the operation of the home charging pile sharing system over a 24-hour period

        Parameters:
            time_steps: Number of discrete time points for the simulation (higher = more granular)

        Returns:
            dict: Simulation results containing time series data of key metrics
        """
        # Generate evenly spaced time points across 24 hours
        simulation_times = np.linspace(0, 24, time_steps, endpoint=False)

        # Initialize dictionary to store simulation results
        simulation_results = {
            'time': simulation_times,
            'arrival_rate': [],
            'effective_capacity': [],
            'dynamic_price': [],
            'avg_waiting_time': []
        }

        # Assume all piles are functional during the simulation (no malfunctions)
        pile_availability = np.ones(self.total_piles)

        # Calculate average service rate (1 / average charging duration)
        avg_charging_duration = self.charging_mu
        average_service_rate = 1 / avg_charging_duration

        # Iterate through each time point to compute metrics
        for t in simulation_times:
            # 1. Compute instantaneous arrival rate
            current_arrival_rate = self.arrival_rate(t)
            simulation_results['arrival_rate'].append(current_arrival_rate)

            # 2. Compute effective shareable capacity
            current_effective_cap = self.effective_capacity(t, pile_availability)
            simulation_results['effective_capacity'].append(current_effective_cap)

            # 3. Compute dynamic price (based on utilization rate)
            if current_effective_cap > 0:
                current_utilization = current_arrival_rate / (current_effective_cap * average_service_rate)
                # Cap utilization at 0.99 to avoid numerical instability near 1
                current_utilization = min(current_utilization, 0.99)
            else:
                current_utilization = 0.0  # No capacity → zero utilization

            current_dynamic_price = self.dynamic_pricing(current_utilization)
            simulation_results['dynamic_price'].append(current_dynamic_price)

            # 4. Compute average waiting time
            current_waiting_time = self.average_waiting_time(t, current_arrival_rate, average_service_rate, pile_availability)
            simulation_results['avg_waiting_time'].append(current_waiting_time)

        return simulation_results


# Example Usage
if __name__ == "__main__":
    # 1. Initialize the home charging pile model with adjusted parameters
    charging_model = HomeChargingPileModel(
        lambda0=30,          # Increased base arrival rate to reflect higher demand
        delta_lambda=20,     # Increased fluctuation to simulate peak/off-peak differences
        p0=2.0,              # Base charging price (e.g., 2 currency units per hour)
        beta=0.2,            # Moderate price sensitivity (0 = no sensitivity, 1 = high sensitivity)
        total_piles=200,     # Total number of home charging piles in the system
        min_price_ratio=0.5  # Minimum price = 50% of base price (prevents underpricing)
    )

    # 2. Run 24-hour simulation (240 time steps = 6-minute granularity)
    daily_simulation = charging_model.simulate_day(time_steps=240)

    # 3. Configure plot style and create subplots
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # Support English/Chinese fonts
    plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Home Charging Pile Sharing Mechanism Simulation Results (24-Hour Period)', fontsize=16, fontweight='bold')

    # 3.1 Plot: Arrival Rate vs. Time
    axes[0, 0].plot(daily_simulation['time'], daily_simulation['arrival_rate'], color='#2E86AB', linewidth=2)
    axes[0, 0].set_title('User Request Arrival Rate Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time (24-Hour Format)', fontsize=10)
    axes[0, 0].set_ylabel('Arrival Rate (Requests per Hour)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 24)

    # 3.2 Plot: Effective Capacity vs. Time
    axes[0, 1].plot(daily_simulation['time'], daily_simulation['effective_capacity'], color='#A23B72', linewidth=2)
    axes[0, 1].set_title('Effective Shareable Capacity Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time (24-Hour Format)', fontsize=10)
    axes[0, 1].set_ylabel('Number of Effective Piles', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 24)

    # 3.3 Plot: Dynamic Price vs. Time
    axes[1, 0].plot(daily_simulation['time'], daily_simulation['dynamic_price'], color='#F18F01', linewidth=2)
    axes[1, 0].set_title('Dynamic Charging Price Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (24-Hour Format)', fontsize=10)
    axes[1, 0].set_ylabel('Charging Price (Currency Units per Hour)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 24)

    # 3.4 Plot: Average Waiting Time vs. Time
    axes[1, 1].plot(daily_simulation['time'], daily_simulation['avg_waiting_time'], color='#C73E1D', linewidth=2)
    axes[1, 1].set_title('Average User Waiting Time Over Time', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time (24-Hour Format)', fontsize=10)
    axes[1, 1].set_ylabel('Average Waiting Time (Hours)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 24)
    # Limit y-axis to avoid extreme values (infinite waiting time)
    axes[1, 1].set_ylim(0, min(5, max(daily_simulation['avg_waiting_time'][1:])))  # Skip first value if infinite

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # 4. Define PDF save path and save the plot
    save_directory = r"E:\0ECJTU\0ESSAY\arxiv\MSO\code_MSO"
    # Create directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)
        print(f"Directory created: {save_directory}")

    # Define full PDF path
    pdf_save_path = os.path.join(save_directory, "home_charging_pile_simulation_results.pdf")
    # Save plot as PDF (high resolution for academic use)
    plt.savefig(pdf_save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()  # Close plot to free memory
    print(f"Simulation plot saved as PDF: {pdf_save_path}")

    # 5. Generate and print charging duration statistics
    charging_duration_samples = charging_model.generate_charging_duration(100)
    print("\n=== Charging Duration Statistics (100 Samples) ===")
    print(f"Mean: {np.mean(charging_duration_samples):.2f} hours")
    print(f"Minimum: {np.min(charging_duration_samples):.2f} hours")
    print(f"Maximum: {np.max(charging_duration_samples):.2f} hours")
    print(f"Standard Deviation: {np.std(charging_duration_samples):.2f} hours")