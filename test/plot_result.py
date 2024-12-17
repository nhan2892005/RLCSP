import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import numpy as np

class ResultVisualizer:
    def __init__(self):
        self.df = pd.read_csv('./test/data/results.csv')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.assets_dir = "./assets/results/_checkpoint_" + self.timestamp
        os.makedirs(self.assets_dir, exist_ok=True)

    def plot_waste_by_policy(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Policy', y='Waste (%)', data=self.df)
        plt.title('Waste Percentage by Policy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.assets_dir}/waste_by_policy.png")
        plt.close()

    def plot_runtime_comparison(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Policy', y='Run times (s)', data=self.df)
        plt.title('Runtime Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.assets_dir}/runtime_comparison.png")
        plt.close()

    def plot_efficiency_scatter(self):
        plt.figure(figsize=(10, 6))
        for policy in self.df['Policy'].unique():
            mask = self.df['Policy'] == policy
            plt.scatter(
                self.df[mask]['Run times (s)'],
                self.df[mask]['Waste (%)'],
                label=policy,
                alpha=0.6
            )
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Waste (%)')
        plt.title('Efficiency: Runtime vs Waste')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.assets_dir}/efficiency_scatter.png")
        plt.close()

    def plot_product_impact(self):
        """Plot impact of number of products on waste percentage"""
        plt.figure(figsize=(12, 8))
        
        # Plot points for each policy with different colors
        for policy in self.df['Policy'].unique():
            mask = self.df['Policy'] == policy
            plt.scatter(
                self.df[mask]['Number of products'],
                self.df[mask]['Waste (%)'],
                label=policy,
                alpha=0.6
            )
            
            # Add trend line for each policy
            z = np.polyfit(
                self.df[mask]['Number of products'], 
                self.df[mask]['Waste (%)'], 
                1
            )
            p = np.poly1d(z)
            x_trend = np.linspace(
                self.df['Number of products'].min(),
                self.df['Number of products'].max(),
                100
            )
            plt.plot(x_trend, p(x_trend), linestyle='--', alpha=0.3)
        
        plt.xlabel('Number of Products')
        plt.ylabel('Waste (%)')
        plt.title('Impact of Product Quantity on Waste Percentage')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
        # Save plot
        plt.savefig(
            f"{self.assets_dir}/product_quantity_impact.png",
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    def plot_stocks_usage(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Policy', y='Number of used stocks', data=self.df)
        plt.title('Number of Stocks Used by Policy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.assets_dir}/stocks_usage.png")
        plt.close()

    def generate_summary(self):
        summary = self.df.groupby('Policy').agg({
            'Run times (s)': ['mean', 'std', 'min', 'max'],
            'Waste (%)': ['mean', 'std', 'min', 'max'],
            'Number of used stocks': ['mean', 'std']
        }).round(3)
        
        summary.to_csv(f"{self.assets_dir}/summary.csv")
        return summary

def main():
    visualizer = ResultVisualizer()
    
    # Generate all plots
    visualizer.plot_waste_by_policy()
    visualizer.plot_runtime_comparison()
    visualizer.plot_efficiency_scatter()
    visualizer.plot_product_impact()
    visualizer.plot_stocks_usage()
    
    print(f"\nSummary Statistics in : {visualizer.assets_dir}/summary.csv")
    print(f"\nPlots saved in: {visualizer.assets_dir}/")

if __name__ == "__main__":
    main()