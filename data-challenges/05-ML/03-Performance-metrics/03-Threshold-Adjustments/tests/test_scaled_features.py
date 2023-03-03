from nbresult import ChallengeResultTestCase


class TestScaled_features(ChallengeResultTestCase):
    def test_scaled_features(self):
        min = self.result.scaled_features.min()
        self.assertLess(min, 10)
        self.assertGreater(min, -10)
