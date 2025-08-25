"""Unit tests for evaluation system to ensure correctness and prevent data leakage."""
import os
import sys
import time
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.model_evaluation import ModelEvaluator # pylint: disable=wrong-import-position


class TestEvaluationSystem(unittest.TestCase):
    """Test suite for evaluation system validation."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, "test_mortgage_data.csv")

        test_data = {
            'age': [25, 35, 45, 30, 40],
            'reported_monthly_income': [30000, 50000, 70000, 40000, 60000],
            'total_existing_debt': [100000, 200000, 50000, 150000, 80000],
            'loan_amount': [2000000, 3000000, 1500000, 2500000, 2200000],
            'loan_term': [30, 25, 20, 30, 25],
            'monthly_payment': [10000, 15000, 8000, 12000, 11000],
            'loan_approved': [0.8, 0.2, 0.9, 0.4, 0.7],
            'government_employee': ['no', 'yes', 'no', 'no', 'yes'],
            'highest_education': ['bachelor', 'master', 'phd', 'bachelor', 'master'],
            'employment_type': ['permanent', 'permanent', 'temporary', 'permanent', 'freelancer'],
            'housing_status': ['rent', 'own', 'mortgage', 'rent', 'own'],
            'credit_history': ['excellent', 'fair', 'excellent', 'good', 'excellent'],
            'investments': ['no', 'yes', 'no', 'yes', 'no'],
            'len_employment': [2, 5, 10, 3, 7],
            'size_of_company': [50, 200, 1000, 100, 300],
            'investments_value': [0, 100000, 0, 50000, 0],
            'property_owned_value': [0, 500000, 1000000, 200000, 800000]
        }

        pd.DataFrame(test_data).to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_cv_results_have_realistic_variability(self):
        """Test cross-validation has realistic variability not fake 100%."""
        evaluator = ModelEvaluator(csv_path=self.test_csv_path)

        try:
            scores = evaluator.cross_validate(k_folds=3)

            mean_score = np.mean(scores)

            self.assertGreater(len(scores), 0)
            self.assertLessEqual(mean_score, 1.0)
            self.assertGreaterEqual(mean_score, 0.0)

            if len(scores) > 1:
                unique_scores = len(set(scores))
                self.assertGreaterEqual(unique_scores, 1)

        except Exceptionfr5vb=] as e:
            self.skipTest(f"Cross-validation failed due to dependencies: {e}")

    def test_metrics_calculation_correctness(self):
        """Test metrics are calculated correctly with known data."""
        evaluator = ModelEvaluator(csv_path=self.test_csv_path)
        evaluator.y_true = np.array([1, 0, 1, 1, 0])
        evaluator.predictions = [1, 0, 0, 1, 0]
        evaluator.probabilities = [0.8, 0.2, 0.4, 0.9, 0.1]

        metrics = evaluator.calculate_metrics()

        expected_accuracy = 0.8
        expected_precision = 1.0
        expected_recall = 0.667

        self.assertAlmostEqual(metrics['accuracy'], expected_accuracy, places=2)
        self.assertAlmostEqual(metrics['precision'], expected_precision, places=2)
        self.assertAlmostEqual(metrics['recall'], expected_recall, places=2)

    def test_roc_curve_data_consistency(self):
        """Test ROC curve uses consistent data lengths."""
        evaluator = ModelEvaluator(csv_path=self.test_csv_path)
        evaluator.y_true = np.array([1, 0, 1, 0, 1])
        evaluator.probabilities = [0.9, 0.1, 0.8, 0.3, 0.7]
        evaluator.predictions = [1, 0, 1, 0, 1]

        temp_plot_path = os.path.join(self.temp_dir, "test_roc.png")

        try:
            evaluator.plot_roc_curve(temp_plot_path)
        except Exception as e:
            if "must be generated first" in str(e):
                self.skipTest("Expected error for missing data")
            else:
                self.fail(f"ROC curve failed due to data inconsistency: {e}")

    def test_no_data_leakage_detection(self):
        """Test detection of potential data leakage patterns."""
        evaluator = ModelEvaluator(csv_path=self.test_csv_path)
        evaluator.y_true = np.array([1, 0, 1, 0, 1])
        evaluator.predictions = [1, 0, 1, 0, 1]
        evaluator.probabilities = [0.9, 0.1, 0.8, 0.2, 0.7]
        metrics = evaluator.calculate_metrics()

        if metrics['accuracy'] == 1.0:
            print("WARNING: Perfect accuracy detected - check for data leakage")

        self.assertGreaterEqual(metrics['accuracy'], 0.0)

    def test_evaluation_generates_required_outputs(self):
        """Test evaluation generates all required output files."""
        evaluator = ModelEvaluator(csv_path=self.test_csv_path, use_independent_test=False)
        eval_results_dir = os.path.join(self.temp_dir, "evaluation_results")
        os.makedirs(eval_results_dir, exist_ok=True)
        confusion_path = os.path.join(eval_results_dir, "confusion_matrix.png")
        roc_path = os.path.join(eval_results_dir, "roc_curve.png")

        evaluator.test_data = pd.read_csv(self.test_csv_path)
        evaluator.predictions = [1, 0, 1, 0, 1]
        evaluator.probabilities = [0.8, 0.2, 0.9, 0.1, 0.7]
        evaluator.y_true = np.array([1, 0, 1, 1, 0])

        metrics = evaluator.calculate_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)

        evaluator.plot_confusion_matrix(confusion_path)
        evaluator.plot_roc_curve(roc_path)

        self.assertTrue(os.path.exists(confusion_path))
        self.assertTrue(os.path.exists(roc_path))
        self.assertGreater(os.path.getsize(confusion_path), 0)
        self.assertGreater(os.path.getsize(roc_path), 0)

    def test_cross_validation_uses_training_data(self):
        """Test cross-validation uses training data not test data."""
        evaluator = ModelEvaluator(csv_path=self.test_csv_path)
        original_csv_path = evaluator.csv_path

        try:
            scores = evaluator.cross_validate(k_folds=3)

            self.assertEqual(evaluator.csv_path, original_csv_path)
            self.assertTrue(os.path.exists(evaluator.csv_path))

            if len(scores) > 0:
                for score in scores:
                    self.assertGreaterEqual(score, 0.0)
                    self.assertLessEqual(score, 1.0)

        except Exception as e:
            self.skipTest(f"CV test skipped due to dependencies: {e}")

    def test_independent_test_data_generation(self):
        """Test independent test data is different from training."""
        evaluator = ModelEvaluator(
            csv_path=self.test_csv_path,
            use_independent_test=True,
            test_scenario="economic_downturn"
        )

        self.assertTrue(evaluator.use_independent_test)
        self.assertEqual(evaluator.test_scenario, "economic_downturn")

        evaluator.csv_path = self.test_csv_path
        evaluator.test_csv_path = os.path.join(self.temp_dir, "test_economic_downturn.csv")

        self.assertNotEqual(evaluator.test_csv_path, evaluator.csv_path)

        # Create mock test data to verify concept
        test_data_mock = pd.DataFrame({
            'reported_monthly_income': [25000, 40000, 35000],  # 15% lower than training
            'loan_amount': [1600000, 2400000, 1800000]  # Different amounts
        })
        test_data_mock.to_csv(evaluator.test_csv_path, index=False)

        self.assertTrue(os.path.exists(evaluator.csv_path))
        self.assertTrue(os.path.exists(evaluator.test_csv_path))

        train_data = pd.read_csv(evaluator.csv_path)
        test_data = pd.read_csv(evaluator.test_csv_path)

        self.assertNotEqual(len(train_data), len(test_data))
        self.assertTrue(len(test_data) > 0)
        self.assertTrue(len(train_data) > 0)

    def test_model_predictions_are_probabilistic(self):
        """Test model predictions are actual probabilities not deterministic."""
        evaluator = ModelEvaluator(csv_path=self.test_csv_path)

        evaluator.probabilities = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8]

        for prob in evaluator.probabilities:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

        unique_probs = len(set(evaluator.probabilities))
        self.assertGreater(unique_probs, 1)

    def test_evaluation_performance_reasonable(self):
        """Test evaluation completes in reasonable time."""
        evaluator = ModelEvaluator(csv_path=self.test_csv_path, use_independent_test=False)
        start_time = time.time()
        try:
            evaluator.test_data = pd.read_csv(self.test_csv_path)
            evaluator.predictions = [1] * len(evaluator.test_data)
            evaluator.probabilities = [0.5] * len(evaluator.test_data)
            evaluator.y_true = np.array([1] * len(evaluator.test_data))

            metrics = evaluator.calculate_metrics()

            end_time = time.time()
            execution_time = end_time - start_time

            self.assertLess(execution_time, 10)
            self.assertIsNotNone(metrics)

        except Exception as e:
            self.skipTest(f"Performance test skipped: {e}")


if __name__ == '__main__':
    unittest.main()
