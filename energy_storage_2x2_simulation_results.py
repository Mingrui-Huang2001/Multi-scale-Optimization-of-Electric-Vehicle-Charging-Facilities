import numpy as np
import math
import matplotlib.pyplot as plt
import os  # Import os module for path handling and directory creation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from typing import List, Tuple, Dict, Union


class EnergyStorageScheduler:
    def __init__(self):
        # Core parameters initialization
        self.soc_states = np.arange(20, 100, 10)  # SOC states: 20%, 30%, ..., 90%
        self.target_soc = 60  # Target SOC value
        self.num_charging_piles = 5  # Number of charging piles

        # Reward function weights
        self.w1 = 0.4  # Weight for cost term
        self.w2 = 0.3  # Weight for grid fluctuation term
        self.w3 = 0.3  # Weight for SOC safety term

        # Baseline values for normalization
        self.C_base = 100.0  # Baseline cost
        self.C_hsae = 120.0  # Reference cost
        self.sigma_basec = 50.0  # Baseline grid fluctuation

        # Initialize dynamic programming value function
        self.value_function = self._initialize_value_function()
        self.reset_history()  # Initialize history storage

    def reset_history(self):
        """Reset history storage for a new simulation run."""
        self.history = {
            'soc': [],
            'queue_length': [],
            'reward': [],
            'lyapunov': [],
            'action': [],
            'electricity_price': [],
            'grid_fluctuation': []
        }

    def _initialize_value_function(self) -> Dict[Tuple[float, int], float]:
        """Initialize value function (all states start at 0)."""
        value_function = {}
        for soc in self.soc_states:
            for q in range(10):  # Upper bound for queue encoding
                value_function[(soc, q)] = 0.0
        return value_function

    def queue_encoding(self, n_queue: int) -> int:
        """Compress queue state: q(t) = floor(log2(N_queue + 1))."""
        if n_queue < 0:
            raise ValueError("Queue length cannot be negative.")
        return int(math.floor(math.log2(n_queue + 1)))

    def reward_function(self, soc: float, current_cost: float, grid_fluctuation: float) -> float:
        """
        Multi-objective reward function:
        r_t = w1·(C_hsae - C_t)/C_base + w2·(σ_t/σ_basec) + w3·I(SOC ∈ [30%, 80%])
        """
        cost_term = self.w1 * (self.C_hsae - current_cost) / self.C_base
        fluctuation_term = self.w2 * (grid_fluctuation / self.sigma_basec)
        safe_term = self.w3 * 1.0 if (30 <= soc <= 80) else self.w3 * (-1.0)
        return cost_term + fluctuation_term + safe_term

    def lyapunov_function(self, soc: float, queue_lengths: List[int]) -> float:
        """
        Lyapunov function for stability:
        L(t) = 0.5 * (Q_bat²(t) + ΣQ_queue,i²(t)), where Q_bat = SOC_target - SOC(t).
        """
        q_bat = self.target_soc - soc
        queue_sum = sum(q ** 2 for q in queue_lengths)
        return 0.5 * (q_bat ** 2 + queue_sum)

    def select_action(self, soc: float, queue_encoding: int, electricity_price: float) -> str:
        """Select action (charge/discharge/do_nothing) based on state and price."""
        soc = max(min(soc, 90), 20)  # Clamp SOC to valid range
        if queue_encoding < 2:  # Low charging demand
            if soc < 50 and electricity_price < 0.8 * self.C_base / self.C_hsae * 100:
                return "charge"
            elif soc > 80:
                return "discharge"
            else:
                return "do_nothing"
        else:  # High charging demand
            return "discharge" if soc > 30 else "charge"

    def dynamic_programming_update(self, state: Tuple[float, int],
                                   reward: float, next_state: Tuple[float, int],
                                   discount_factor: float = 0.9) -> None:
        """Update value function via Bellman equation."""
        self.value_function[state] = reward + discount_factor * self.value_function[next_state]

    def simulate_step(self, current_soc: float, current_queue: int,
                      electricity_price: float, grid_fluctuation: float) -> Tuple[float, int, str]:
        """Simulate one time step and record history."""
        q_encoded = self.queue_encoding(current_queue)
        reward = self.reward_function(current_soc, electricity_price, grid_fluctuation)
        action = self.select_action(current_soc, q_encoded, electricity_price)

        # Update SOC based on action
        if action == "charge":
            next_soc = min(current_soc + 10, 90)
        elif action == "discharge":
            next_soc = max(current_soc - 10, 20)
        else:
            next_soc = current_soc

        # Simulate queue dynamics
        next_queue = max(0, current_queue + np.random.randint(-2, 3))

        # Update value function
        self.dynamic_programming_update((current_soc, q_encoded),
                                        reward,
                                        (next_soc, self.queue_encoding(next_queue)))

        # Calculate Lyapunov function
        queue_lengths = [current_queue] * self.num_charging_piles
        lyapunov = self.lyapunov_function(current_soc, queue_lengths)

        # Log history
        self.history['soc'].append(current_soc)
        self.history['queue_length'].append(current_queue)
        self.history['reward'].append(reward)
        self.history['lyapunov'].append(lyapunov)
        self.history['action'].append(action)
        self.history['electricity_price'].append(electricity_price)
        self.history['grid_fluctuation'].append(grid_fluctuation)

        return next_soc, next_queue, action

    def run_simulation(self, steps: int = 30, initial_soc: float = 50.0, initial_queue: int = 3) -> None:
        """Run a complete simulation with specified steps and initial conditions."""
        self.reset_history()
        current_soc, current_queue = initial_soc, initial_queue
        for _ in range(steps):
            electricity_price = np.random.uniform(70, 110)
            grid_fluctuation = np.random.uniform(20, 60)
            current_soc, current_queue, _ = self.simulate_step(
                current_soc, current_queue, electricity_price, grid_fluctuation
            )

    def plot_single_simulation(self, ax_group, sim_idx: int) -> None:
        """Plot results for a single simulation into a group of subplots."""
        ax_soc, ax_queue_price, ax_reward, ax_lyapunov = ax_group

        # 1. SOC Variation Plot
        ax_soc.plot(self.history['soc'], 'b-', linewidth=1.5)
        ax_soc.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Safe Lower Bound (30%)')
        ax_soc.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Safe Upper Bound (80%)')
        ax_soc.axhline(y=self.target_soc, color='g', linestyle='-', alpha=0.5, label=f'Target SOC ({self.target_soc}%)')
        ax_soc.set_title(f'Sim {sim_idx + 1}: SOC', fontsize=10)
        ax_soc.set_xlabel('Time Step', fontsize=8)
        ax_soc.set_ylabel('SOC (%)', fontsize=8)
        ax_soc.grid(alpha=0.3)
        ax_soc.legend(fontsize=8)
        ax_soc.tick_params(labelsize=7)

        # 2. Queue Length & Electricity Price Plot
        ax_queue_price.plot(self.history['queue_length'], 'r-', linewidth=1.5, label='Queue Length')
        ax_queue_price.set_ylabel('Queue Length', color='r', fontsize=8)
        ax_queue_price.tick_params(axis='y', labelcolor='r', labelsize=7)
        ax_queue_price.set_title(f'Sim {sim_idx + 1}: Queue & Price', fontsize=10)
        ax_queue_price.set_xlabel('Time Step', fontsize=8)
        ax_queue_price.grid(alpha=0.3)

        ax_queue_price_twin = ax_queue_price.twinx()
        ax_queue_price_twin.plot(self.history['electricity_price'], 'g--', linewidth=1.5, label='Electricity Price')
        ax_queue_price_twin.set_ylabel('Electricity Price', color='g', fontsize=8)
        ax_queue_price_twin.tick_params(axis='y', labelcolor='g', labelsize=7)

        # Combine legends
        lines1, labels1 = ax_queue_price.get_legend_handles_labels()
        lines2, labels2 = ax_queue_price_twin.get_legend_handles_labels()
        ax_queue_price.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        ax_queue_price.tick_params(labelsize=7)
        ax_queue_price_twin.tick_params(labelsize=7)

        # 3. Reward Function Plot
        ax_reward.plot(self.history['reward'], 'm-', linewidth=1.5)
        ax_reward.set_title(f'Sim {sim_idx + 1}: Reward', fontsize=10)
        ax_reward.set_xlabel('Time Step', fontsize=8)
        ax_reward.set_ylabel('Reward Value', fontsize=8)
        ax_reward.grid(alpha=0.3)
        ax_reward.tick_params(labelsize=7)

        # 4. Lyapunov Function (Stability) Plot
        ax_lyapunov.plot(self.history['lyapunov'], 'c-', linewidth=1.5)
        ax_lyapunov.set_title(f'Sim {sim_idx + 1}: Lyapunov', fontsize=10)
        ax_lyapunov.set_xlabel('Time Step', fontsize=8)
        ax_lyapunov.set_ylabel('Lyapunov Value', fontsize=8)
        ax_lyapunov.grid(alpha=0.3)
        ax_lyapunov.tick_params(labelsize=7)


# Run 4 random simulations and save result as PDF to specified path
if __name__ == "__main__":
    # -------------------------- Matplotlib Configuration for PDF --------------------------
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 8
    plt.rcParams["figure.dpi"] = 300  # High DPI for print-quality PDF
    plt.rcParams["savefig.dpi"] = 300  # Ensure saved PDF uses high DPI
    plt.rcParams["pdf.fonttype"] = 42  # Avoid font embedding issues (compatible with all readers)
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["savefig.facecolor"] = "white"  # White background (avoids transparent issues)

    # -------------------------- PDF Save Path Configuration --------------------------
    # Target save directory (as specified by user)
    save_directory = r"E:\0ECJTU\0ESSAY\arxiv\MSO\code_MSO"
    # PDF file name (descriptive for easy identification)
    pdf_filename = "energy_storage_2x2_simulation_results.pdf"
    # Full path: combine directory and filename (handles Windows path separators automatically)
    full_pdf_path = os.path.join(save_directory, pdf_filename)

    # Create directory if it doesn't exist (avoids "file not found" errors)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)  # exist_ok=True prevents errors if dir exists
        print(f"Created target directory: {save_directory}")
    else:
        print(f"Using existing directory: {save_directory}")

    # -------------------------- Run Simulations --------------------------
    num_simulations = 4  # 2x2 grid requires 4 simulations
    all_schedulers = []

    print("\nRunning simulations...")
    for i in range(num_simulations):
        scheduler = EnergyStorageScheduler()
        scheduler.run_simulation(steps=30, initial_soc=50.0, initial_queue=3)
        all_schedulers.append(scheduler)
        print(f"Completed simulation {i + 1}/{num_simulations}")

    # -------------------------- Create 2x2 Plot Layout --------------------------
    fig_time_series = plt.figure(figsize=(12, 12))  # Square layout for balanced 2x2 grid
    # Outer grid with increased spacing to prevent overlap
    gs_time_series = GridSpec(2, 2, figure=fig_time_series, wspace=0.4, hspace=0.5)
    ax_groups = []

    # Create inner 2x2 subplot groups for each simulation
    for row in range(2):
        for col in range(2):
            gs_sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_time_series[row, col],
                                             wspace=0.3, hspace=0.4)
            ax_soc = fig_time_series.add_subplot(gs_sub[0, 0])
            ax_queue_price = fig_time_series.add_subplot(gs_sub[0, 1])
            ax_reward = fig_time_series.add_subplot(gs_sub[1, 0])
            ax_lyapunov = fig_time_series.add_subplot(gs_sub[1, 1])
            ax_groups.append([ax_soc, ax_queue_price, ax_reward, ax_lyapunov])

    # Populate plots with simulation data
    for i, scheduler in enumerate(all_schedulers):
        scheduler.plot_single_simulation(ax_groups[i], i)

    # Add main title (adjust y-position to avoid overlap with subplots)
    fig_time_series.suptitle('4 Random Simulations: Energy Storage Scheduling Metrics',
                           fontsize=14, y=0.98)

    # -------------------------- Adjust Layout and Save PDF --------------------------
    # Fine-tune layout to ensure all elements fit in PDF
    plt.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.06,
                        wspace=0.4, hspace=0.5)

    # Save plot as PDF (bbox_inches='tight' prevents edge elements from being cut off)
    plt.savefig(
        full_pdf_path,
        format="pdf",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
        edgecolor="none"
    )

    # Close figure to free memory
    plt.close(fig_time_series)

    # Confirm save success
    if os.path.exists(full_pdf_path):
        file_size = os.path.getsize(full_pdf_path) / 1024 / 1024  # Convert to MB
        print(f"\nPDF saved successfully!")
        print(f"Path: {full_pdf_path}")
        print(f"File size: {file_size:.2f} MB")
    else:
        print(f"\nError: Failed to save PDF to {full_pdf_path}")