from nbresult import ChallengeResultTestCase


class TestClassification(ChallengeResultTestCase):
    def test_change(self):
        score_original = self.result.score_original
        time_original = self.result.time_original
        score_reduced = self.result.score_reduced
        time_reduced = self.result.time_reduced
        change_score = 1 - score_reduced / score_original
        change_time = 1 - time_reduced / time_original
        self.assertLessEqual(change_score, 1)
        self.assertGreaterEqual(change_time, .8)