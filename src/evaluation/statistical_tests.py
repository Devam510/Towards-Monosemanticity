import sys
import os
from pathlib import Path
import torch
import scipy.stats

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def calculate_mann_whitney(ccjfr_scores: list, baseline_scores: list) -> float:
    """
    Computes p-value to determine if CCJFR is statistically better than baseline
    across a set of independent measurements using the non-parametric Mann-Whitney U test.
    """
    # alternative='greater' tests if ccjfr_scores > baseline_scores
    statistic, p_value = scipy.stats.mannwhitneyu(ccjfr_scores, baseline_scores, alternative='greater')
    return p_value

def calculate_mcnemars(ccjfr_successes: list, baseline_successes: list) -> float:
    """
    Calculates McNemar's test for paired categorical data (success/failure on the exact same feature).
    Expects lists of booleans.
    """
    # Contingency table
    #            CCJFR True  | CCJFR False
    # BL True  |      a      |      b
    # BL False |      c      |      d
    
    a = b = c = d = 0
    for c_succ, b_succ in zip(ccjfr_successes, baseline_successes):
        if b_succ and c_succ: a += 1
        elif b_succ and not c_succ: b += 1
        elif not b_succ and c_succ: c += 1
        else: d += 1
        
    # McNemar statistic: (b - c)^2 / (b + c)
    if b + c == 0:
        return 1.0 # no difference
        
    statistic = ((abs(b - c) - 1)**2) / (b + c)
    
    # Chi-square with 1 dof to p-value
    p_value = 1.0 - scipy.stats.distributions.chi2.cdf(statistic, 1)
    return p_value
