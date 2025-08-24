#!/usr/bin/env python3
"""Test script to demonstrate quantitative model evaluation."""

import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_evaluation import ModelEvaluator
from utils.constants import S_GREEN, E_GREEN

def main():
    """Demonstrate quantitative evaluation of the mortgage approval model."""

    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(
            csv_path="datasets/mortgage_applications.csv",
            avg_salary=35000
        )
        
        # Generate comprehensive evaluation report
        metrics = evaluator.generate_evaluation_report()
        
        print(f"\n{S_GREEN}EVALUATION COMPLETED SUCCESSFULLY!{E_GREEN}")
        print("\nKey Results:")
        print(f"   • Accuracy:  {metrics['accuracy']:.3f}")
        print(f"   • Precision: {metrics['precision']:.3f}")
        print(f"   • Recall:    {metrics['recall']:.3f}")
        print(f"   • F1-Score:  {metrics['f1']:.3f}")
        print(f"   • ROC AUC:   {metrics['auc']:.3f}")
        
        print("\nGenerated Files:")
        print("   • evaluation_results/confusion_matrix.png")
        print("   • evaluation_results/roc_curve.png") 
        print("   • evaluation_results/metrics_summary.txt\n")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)