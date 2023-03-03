from nbresult import ChallengeResultTestCase


class TestCvResults(ChallengeResultTestCase):
    def test_cv_results(self):
        self.assertEqual(len(self.result.cv_result['test_score']), 5)

    def test_cv_min(self):
        self.assertEqual(
            self.result.min_score, self.result.cv_result['test_score'].min())

    def test_cv_max(self):
        self.assertEqual(
            self.result.max_score, self.result.cv_result['test_score'].max())

    def test_cv_mean(self):
        self.assertEqual(
            self.result.mean_score, self.result.cv_result['test_score'].mean())
