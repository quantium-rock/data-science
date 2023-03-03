from nbresult import ChallengeResultTestCase


class TestTrainedModel(ChallengeResultTestCase):
    def test_slope(self):
        self.assertEqual(self.result.slope, self.result.model.coef_)

    def test_intercept(self):
        self.assertEqual(self.result.intercept, self.result.model.intercept_)
