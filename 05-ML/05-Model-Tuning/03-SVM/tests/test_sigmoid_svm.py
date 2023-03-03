from nbresult import ChallengeResultTestCase


class TestSigmoidSvm(ChallengeResultTestCase):

    def test_accuracy(self):

        self.assertGreaterEqual(self.result.sigmoid_svm_cv_accuracy, 0.9)
