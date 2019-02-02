import unittest
from examples import bow, fasttext_idf, skip_thought


class TestWorking(unittest.TestCase):

    def test_bow(self):
        self.assertTrue(bow.check() != "")

    def test_fasttext_idf(self):
        self.assertTrue(fasttext_idf.check() != "")

    def test_skip_thought(self):
        self.assertTrue(skip_thought.check() != "")


if __name__ == '__main__':
    unittest.main()
