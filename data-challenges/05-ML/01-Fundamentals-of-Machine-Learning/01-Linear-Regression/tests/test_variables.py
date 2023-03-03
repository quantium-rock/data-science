from nbresult import ChallengeResultTestCase


class TestVariables(ChallengeResultTestCase):
    def test_variable_X(self):
        self.assertEqual(self.result.variable_X, 1525.0)

    def test_variable_y(self):
        self.assertEqual(self.result.variable_y, 182290.0)
