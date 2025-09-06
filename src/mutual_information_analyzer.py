import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union


class MutualInformationAnalyzer:
    """
    Compute mutual information between experimental scores from two groups.
    Handles continuous scores by discretization and provides multiple MI methods.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def compute_mi_discrete(self, scores1: np.ndarray, scores2: np.ndarray, 
                           n_bins: int = 5, binning_method: str = 'uniform') -> float:
        """
        Compute MI by discretizing continuous scores into bins.
        
        Args:
            scores1, scores2: Arrays of scores for each group
            n_bins: Number of bins for discretization
            binning_method: 'uniform', 'quantile', or 'kmeans'
        
        Returns:
            Mutual information value
        """
        # Discretize the scores
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', 
                                     strategy=binning_method, 
                                     random_state=self.random_state)
        
        # Fit on combined data to ensure consistent binning
        combined_data = np.concatenate([scores1, scores2]).reshape(-1, 1)
        discretizer.fit(combined_data)
        
        # Transform both score sets
        disc_scores1 = discretizer.transform(scores1.reshape(-1, 1)).flatten().astype(int)
        disc_scores2 = discretizer.transform(scores2.reshape(-1, 1)).flatten().astype(int)
        
        # Compute mutual information
        mi = mutual_info_score(disc_scores1, disc_scores2)
        
        return mi
    
    def compute_mi_continuous(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """
        Compute MI for continuous variables using sklearn's mutual_info_regression.
        
        Args:
            scores1, scores2: Arrays of scores for each group
            
        Returns:
            Mutual information value
        """
        # Use mutual_info_regression (treats one as continuous target)
        mi = mutual_info_regression(scores1.reshape(-1, 1), scores2, 
                                   random_state=self.random_state)[0]
        return mi
    
    def compute_mi_kde(self, scores1: np.ndarray, scores2: np.ndarray, 
                      n_neighbors: int = 3) -> float:
        """
        Compute MI using KDE-based estimation.
        
        Args:
            scores1, scores2: Arrays of scores for each group
            n_neighbors: Number of neighbors for KDE estimation
            
        Returns:
            Mutual information value
        """
        try:
            from sklearn.feature_selection import mutual_info_regression
            # Use sklearn's KDE-based approach
            mi = mutual_info_regression(scores1.reshape(-1, 1), scores2, 
                                       discrete_features=False,
                                       n_neighbors=n_neighbors,
                                       random_state=self.random_state)[0]
            return mi
        except Exception as e:
            print(f"KDE estimation failed: {e}")
            return self.compute_mi_continuous(scores1, scores2)
    
    def compute_correlation_mi(self, scores1: np.ndarray, scores2: np.ndarray) -> Tuple[float, float]:
        """
        Compute both MI and Pearson correlation for comparison.
        
        Returns:
            Tuple of (mutual_information, pearson_correlation)
        """
        mi = self.compute_mi_continuous(scores1, scores2)
        correlation, _ = stats.pearsonr(scores1, scores2)
        return mi, correlation
    
    def bootstrap_mi_confidence(self, scores1: np.ndarray, scores2: np.ndarray, 
                              n_bootstrap: int = 1000, 
                              method: str = 'continuous',
                              n_bins: int = 5,
                              binning_method: str = 'uniform') -> Tuple[float, float, float]:
        """
        Compute MI with bootstrap confidence intervals.
        
        Args:
            scores1, scores2: Arrays of scores
            n_bootstrap: Number of bootstrap samples
            method: 'continuous', 'discrete', or 'kde'
            n_bins: Number of bins if method is 'discrete'
            binning_method: Binning strategy if method is 'discrete'
            
        Returns:
            Tuple of (mi_estimate, ci_lower, ci_upper)
        """
        n_samples = len(scores1)
        bootstrap_mis = []
        
        # Choose MI computation method
        if method == 'continuous':
            mi_func = self.compute_mi_continuous
        elif method == 'discrete':
            mi_func = lambda x, y: self.compute_mi_discrete(x, y, n_bins=n_bins, binning_method=binning_method)
        elif method == 'kde':
            mi_func = self.compute_mi_kde
        else:
            raise ValueError("Method must be 'continuous', 'discrete', or 'kde'")
        
        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_scores1 = scores1[indices]
            boot_scores2 = scores2[indices]
            
            # Compute MI for bootstrap sample
            boot_mi = mi_func(boot_scores1, boot_scores2)
            bootstrap_mis.append(boot_mi)
        
        # Compute confidence intervals
        mi_estimate = mi_func(scores1, scores2)
        ci_lower = np.percentile(bootstrap_mis, 2.5)
        ci_upper = np.percentile(bootstrap_mis, 97.5)
        # Standard error is the std of bootstrap estimates
        sem = np.std(bootstrap_mis)
        return mi_estimate, ci_lower, ci_upper, sem
    
    def analyze_binning_sensitivity(self, scores1: np.ndarray, scores2: np.ndarray,
                                   bin_range: range = range(3, 21)) -> pd.DataFrame:
        """
        Analyze how MI estimates change with number of bins.
        
        Returns:
            DataFrame with bin counts and corresponding MI values
        """
        results = []
        
        for n_bins in bin_range:
            for method in ['uniform', 'quantile', 'kmeans']:
                try:
                    mi = self.compute_mi_discrete(scores1, scores2, 
                                                n_bins=n_bins, 
                                                binning_method=method)
                    results.append({
                        'n_bins': n_bins,
                        'binning_method': method,
                        'mutual_information': mi
                    })
                except:
                    continue
        
        return pd.DataFrame(results)
    
    def permutation_test(self, scores1: np.ndarray, scores2: np.ndarray,
                        n_permutations: int = 1000,
                        method: str = 'continuous', 
                        n_bins: int = 5, 
                        binning_method: str = 'uniform') -> Tuple[float, float]:
        """
        Test significance of MI using permutation test.
        
        Returns:
            Tuple of (observed_mi, p_value)
        """
        # Compute observed MI
        if method == 'continuous':
            observed_mi = self.compute_mi_continuous(scores1, scores2)
        elif method == 'discrete':
            observed_mi = self.compute_mi_discrete(scores1, scores2, n_bins=n_bins, binning_method=binning_method)
        else:
            observed_mi = self.compute_mi_kde(scores1, scores2)
        
        # Permutation test
        permuted_mis = []
        for _ in range(n_permutations):
            # Shuffle one of the arrays
            shuffled_scores2 = np.random.permutation(scores2)
            
            if method == 'continuous':
                perm_mi = self.compute_mi_continuous(scores1, shuffled_scores2)
            elif method == 'discrete':
                perm_mi = self.compute_mi_discrete(scores1, shuffled_scores2)
            else:
                perm_mi = self.compute_mi_kde(scores1, shuffled_scores2)
            
            permuted_mis.append(perm_mi)
        
        # Calculate p-value
        p_value = np.sum(np.array(permuted_mis) >= observed_mi) / n_permutations
        
        return observed_mi, p_value
    
    def plot_binning_analysis(self, scores1: np.ndarray, scores2: np.ndarray,
                             bin_range: range = range(3, 21)) -> plt.Figure:
        """
        Plot how MI varies with binning strategy and number of bins.
        """
        df = self.analyze_binning_sensitivity(scores1, scores2, bin_range)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot MI vs number of bins for different methods
        for method in df['binning_method'].unique():
            method_data = df[df['binning_method'] == method]
            ax1.plot(method_data['n_bins'], method_data['mutual_information'], 
                    marker='o', label=method)
        
        ax1.set_xlabel('Number of Bins')
        ax1.set_ylabel('Mutual Information')
        ax1.set_title('MI Sensitivity to Binning Strategy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot of original scores
        ax2.scatter(scores1, scores2, alpha=0.6)
        ax2.set_xlabel('Group 1 Scores')
        ax2.set_ylabel('Group 2 Scores')
        ax2.set_title('Score Relationship')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation info
        corr, _ = stats.pearsonr(scores1, scores2)
        ax2.text(0.05, 0.95, f'Pearson r = {corr:.3f}', 
                transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        plt.tight_layout()
        return fig


def comprehensive_mi_analysis(scores1: np.ndarray, scores2: np.ndarray,
                            condition_names: Optional[List[str]] = None) -> dict:
    """
    Comprehensive mutual information analysis between two score sets.
    
    Args:
        scores1, scores2: Arrays of experimental scores
        condition_names: Optional names for the conditions
        
    Returns:
        Dictionary with comprehensive MI analysis results
    """
    analyzer = MutualInformationAnalyzer()
    
    # Basic statistics
    basic_stats = {
        'group1_mean': np.mean(scores1),
        'group1_std': np.std(scores1),
        'group2_mean': np.mean(scores2),
        'group2_std': np.std(scores2),
        'pearson_correlation': stats.pearsonr(scores1, scores2)[0],
        'spearman_correlation': stats.spearmanr(scores1, scores2)[0]
    }
    
    # Multiple MI estimates
    mi_continuous = analyzer.compute_mi_continuous(scores1, scores2)
    mi_discrete_uniform = analyzer.compute_mi_discrete(scores1, scores2, n_bins=5, binning_method='uniform')
    mi_discrete_quantile = analyzer.compute_mi_discrete(scores1, scores2, n_bins=5, binning_method='quantile')
    mi_kde = analyzer.compute_mi_kde(scores1, scores2)
    
    # Bootstrap confidence intervals
    mi_boot, ci_low, ci_high, sem = analyzer.bootstrap_mi_confidence(scores1, scores2, method='continuous')
    
    # Permutation test
    mi_obs, p_value = analyzer.permutation_test(scores1, scores2, method='continuous')
    
    # Normalized MI (0-1 scale)
    # MI normalized by min of individual entropies
    h1 = stats.entropy(np.histogram(scores1, bins=10)[0] + 1e-10)  # Add small constant to avoid log(0)
    h2 = stats.entropy(np.histogram(scores2, bins=10)[0] + 1e-10)
    normalized_mi = mi_continuous / min(h1, h2) if min(h1, h2) > 0 else 0
    
    results = {
        'basic_statistics': basic_stats,
        'mutual_information': {
            'continuous': mi_continuous,
            'discrete_uniform': mi_discrete_uniform,
            'discrete_quantile': mi_discrete_quantile,
            'kde_estimate': mi_kde,
            'normalized_mi': normalized_mi
        },
        'confidence_intervals': {
            'mi_estimate': mi_boot,
            'ci_lower': ci_low,
            'ci_upper': ci_high
        },
        'significance_test': {
            'observed_mi': mi_obs,
            'p_value': p_value
        }
    }
    
    return results


# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    n_conditions = 20
    
    # Create correlated scores with some noise
    base_scores = np.random.normal(0.7, 0.15, n_conditions)
    group1_scores = base_scores + np.random.normal(0, 0.05, n_conditions)
    group2_scores = 0.8 * base_scores + np.random.normal(0, 0.08, n_conditions)
    
    # Ensure scores are in reasonable range (e.g., 0-1 for accuracy)
    group1_scores = np.clip(group1_scores, 0, 1)
    group2_scores = np.clip(group2_scores, 0, 1)
    
    print("Example: Mutual Information Analysis")
    print("=" * 40)
    
    # Comprehensive analysis
    results = comprehensive_mi_analysis(group1_scores, group2_scores, 
                                      condition_names=['Human', 'Model'])
    
    # Print results
    print("Basic Statistics:")
    for key, value in results['basic_statistics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nMutual Information Estimates:")
    for key, value in results['mutual_information'].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nConfidence Intervals (95%):")
    print(f"  MI estimate: {results['confidence_intervals']['mi_estimate']:.4f}")
    print(f"  CI: [{results['confidence_intervals']['ci_lower']:.4f}, "
          f"{results['confidence_intervals']['ci_upper']:.4f}]")
    
    print(f"\nSignificance Test:")
    print(f"  Observed MI: {results['significance_test']['observed_mi']:.4f}")
    print(f"  p-value: {results['significance_test']['p_value']:.4f}")
    
    # Individual analyzer for more detailed analysis
    analyzer = MutualInformationAnalyzer()
    
    # Binning sensitivity analysis
    print(f"\nBinning Sensitivity Analysis:")
    binning_df = analyzer.analyze_binning_sensitivity(group1_scores, group2_scores)
    print(binning_df.groupby('binning_method')['mutual_information'].agg(['mean', 'std']))