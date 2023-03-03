from nbresult import ChallengeResultTestCase


class TestAddedFeatures(ChallengeResultTestCase):
    def test_increased_score(self):
        self.assertGreater(self.result.score, 0.6)
        self.assertLess(self.result.score, 0.8)
