from nbresult import ChallengeResultTestCase

class TestCollinearity(ChallengeResultTestCase):
    def test_removed_highly_correlated_features(self):
        self.assertEqual(len(self.result.dataset.columns) <=11, True)

