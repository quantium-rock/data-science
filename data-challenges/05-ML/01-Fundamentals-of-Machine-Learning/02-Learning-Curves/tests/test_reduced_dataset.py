from nbresult import ChallengeResultTestCase


class TestReducedDataset(ChallengeResultTestCase):
    def test_reduced_dataset_size(self):
        self.assertEqual(len(self.result.dataset), 300)

    def test_reduced_dataset_score(self):
        self.assertGreater(self.result.score, 0.6)
