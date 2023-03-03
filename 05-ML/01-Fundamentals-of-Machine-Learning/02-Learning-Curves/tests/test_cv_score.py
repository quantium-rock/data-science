from nbresult import ChallengeResultTestCase


class TestCvScore(ChallengeResultTestCase):
    def test_cv_score(self):
        self.assertGreater(self.result.score, 0.3)
        self.assertLess(self.result.score, 0.7)
