from nbresult import ChallengeResultTestCase


class TestPrediction(ChallengeResultTestCase):
    def test_prediction_around_250000(self):
        self.assertGreater(self.result.prediction, 150000)
        self.assertLess(self.result.prediction, 350000)
