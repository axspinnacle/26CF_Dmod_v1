import numpy as np
import matplotlib.pyplot as plt


class FoldDiagnostics:
    def show_zero_exposure_stats(self, fold_creator):
        df = fold_creator.df
        total = len(df)
        print(f"Total records: {total:,}")
        print("Zero exposure counts by coverage:")
        for cov in fold_creator.all_coverages:
            col = fold_creator.ee_columns[cov]
            zero_count = (df[col] == 0).sum()
            pct = 100 * zero_count / total
            print(f"  {cov.upper():5s} ({col}): {zero_count:,} zeros ({pct:.1f}%)")

    def plot_objective_distribution(self, results_df):
        objectives = results_df["avg_objective"]
        print(f"Min: {objectives.min():.4f}  Max: {objectives.max():.4f}  Mean: {objectives.mean():.4f}")

        plt.figure(figsize=(10, 6))
        plt.hist(objectives, bins=20, edgecolor="black", alpha=0.7)
        plt.axvline(objectives.min(), color="red", linestyle="--", label=f"Best: {objectives.min():.4f}")
        plt.axvline(objectives.mean(), color="green", linestyle="--", label=f"Mean: {objectives.mean():.4f}")
        plt.xlabel("Average Objective Function Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Objective Function Values Across Seeds")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_all_coverage_histograms(self, fold_creator):
        df = fold_creator.df
        coverages = fold_creator.all_coverages
        n_cols = 3
        n_rows = (len(coverages) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, cov in enumerate(coverages):
            ax = axes[i]
            col = fold_creator.ee_columns[cov]
            data = df[col]
            zero_pct = 100 * (data == 0).sum() / len(data)
            nonzero = data[data > 0]

            if len(nonzero) > 0:
                ax.hist(nonzero, bins=50, edgecolor="black", alpha=0.7)
                ax.set_title(f"{cov.upper()} (Zero: {zero_pct:.1f}%)\nmean={nonzero.mean():.2f}")
            else:
                ax.text(0.5, 0.5, "100% Zero", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{cov.upper()} - All Zero!")
            ax.set_xlabel(col)

        for i in range(len(coverages), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Earned Exposure Distributions (Non-Zero)")
        plt.tight_layout()
        plt.show()

    def plot_incurred_histograms(self, fold_creator):
        df = fold_creator.df
        coverages = fold_creator.all_coverages
        n_cols = 3
        n_rows = (len(coverages) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, cov in enumerate(coverages):
            ax = axes[i]
            col = fold_creator.incurred_columns[cov]
            data = df[col]
            zero_pct = 100 * (data == 0).sum() / len(data)
            nonzero = data[data > 0]

            if len(nonzero) > 0:
                ax.hist(nonzero, bins=50, edgecolor="black", alpha=0.7, color="orange")
                ax.set_title(f"{cov.upper()} (Zero: {zero_pct:.1f}%)\nmean={nonzero.mean():.2f}")
            else:
                ax.text(0.5, 0.5, "100% Zero", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{cov.upper()} - All Zero!")
            ax.set_xlabel(col)

        for i in range(len(coverages), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Incurred Loss Distributions (Non-Zero)")
        plt.tight_layout()
        plt.show()

    def analyze_bi_pd_relationship(self, fold_creator):
        df = fold_creator.df
        ee_bi = fold_creator.ee_columns["bi"]
        ee_pd = fold_creator.ee_columns["pd"]
        inc_bi = fold_creator.incurred_columns["bi"]
        inc_pd = fold_creator.incurred_columns["pd"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.scatter(df[ee_bi], df[ee_pd], alpha=0.1, s=1)
        ax.set_xlabel(f"BI Exposure ({ee_bi})")
        ax.set_ylabel(f"PD Exposure ({ee_pd})")
        ax.set_title("BI vs PD Exposure")
        max_val = max(df[ee_bi].max(), df[ee_pd].max())
        ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y=x")
        ax.legend()

        ax = axes[0, 1]
        ee_diff = df[ee_bi] - df[ee_pd]
        ax.hist(ee_diff, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--")
        ax.axvline(ee_diff.mean(), color="green", linestyle="--", label=f"Mean: {ee_diff.mean():.4f}")
        ax.set_xlabel("BI Exposure - PD Exposure")
        ax.set_title("Exposure Difference (BI - PD)")
        ax.legend()

        ax = axes[1, 0]
        combined_inc = df[inc_bi] + df[inc_pd]
        nonzero = combined_inc[combined_inc > 0]
        if len(nonzero) > 0:
            ax.hist(nonzero, bins=50, edgecolor="black", alpha=0.7, color="purple")
            ax.set_title(f"BI+PD Combined Incurred\nmean={nonzero.mean():.2f}")
        ax.set_xlabel("BI + PD Incurred Loss")

        ax = axes[1, 1]
        bi_pp = (df[inc_bi] / df[ee_bi]).replace([np.inf, -np.inf], np.nan)
        pd_pp = (df[inc_pd] / df[ee_pd]).replace([np.inf, -np.inf], np.nan)
        ax.hist(bi_pp.dropna(), bins=50, alpha=0.5, label=f"BI PP (mean={bi_pp.mean():.2f})", edgecolor="black")
        ax.hist(pd_pp.dropna(), bins=50, alpha=0.5, label=f"PD PP (mean={pd_pp.mean():.2f})", edgecolor="black")
        ax.set_xlabel("Pure Premium")
        ax.set_title("BI vs PD Pure Premium Comparison")
        ax.legend()

        plt.tight_layout()
        plt.show()

        corr = df[ee_bi].corr(df[ee_pd])
        same_pct = 100 * (ee_diff == 0).sum() / len(df)
        print(f"Correlation: {corr:.4f}, same_pct: {same_pct:.1f}%")
        if same_pct > 90:
            print("BI and PD nearly identical - treat as same exposure")
        elif corr > 0.95:
            print("BI and PD highly correlated - could combine")
        else:
            print("BI and PD moderately correlated - treat separately")
