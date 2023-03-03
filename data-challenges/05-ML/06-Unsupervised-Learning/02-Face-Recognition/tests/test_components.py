from nbresult import ChallengeResultTestCase


class TestComponents(ChallengeResultTestCase):
    def test_minimal_pc(self):
        self.assertEqual(self.result.min_pc, 30)
