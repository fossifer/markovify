import unittest
import markovify
import sys, os
import operator

def get_sorted(chain_json):
    return sorted(chain_json, key=operator.itemgetter(0))

with open(os.path.join(os.path.dirname(__file__), "texts/sherlock.txt")) as f:
    sherlock = f.read()
    sherlock_model = markovify.Text(sherlock)
    sherlock_model_no_retain = markovify.Text(sherlock, retain_original=False)
    sherlock_model_compiled = sherlock_model.compile()

class MarkovifyTest(unittest.TestCase):

    def test_double_weighted(self):
        text_model = sherlock_model
        combo = markovify.append(text_model, [ text_model ])
        assert(combo.chain.model != text_model.chain.model)

    def test_append_chains(self):
        chain = sherlock_model.chain
        combo = markovify.append(chain, [ chain ])

    def test_append_dicts(self):
        _dict = sherlock_model.chain.model
        _dict_r = sherlock_model.chain.model_reversed
        
        combo = markovify.append([ _dict, _dict_r], [[ _dict, _dict_r ]])

    def test_append_lists(self):
        _list = list(sherlock_model.chain.model.items())
        _list_r = list(sherlock_model.chain.model_reversed.items())
        combo = markovify.append([ _list, _list_r ], ([ _list, _list_r ]))

    def test_bad_types(self):
        with self.assertRaises(Exception) as context:
            combo = markovify.append("testing", [ "testing" ])

    def test_bad_weights(self):
        with self.assertRaises(Exception) as context:
            text_model = sherlock_model
            combo = markovify.append(text_model, [ text_model ], [ 0.5, 0.5 ])

    def test_mismatched_state_sizes(self):
        with self.assertRaises(Exception) as context:
            text_model_a = markovify.Text(sherlock, state_size=2)
            text_model_b = markovify.Text(sherlock, state_size=3)
            combo = markovify.append(text_model_a, [ text_model_b ])

    def test_mismatched_model_types(self):
        with self.assertRaises(Exception) as context:
            text_model_a = sherlock_model
            text_model_b = markovify.NewlineText(sherlock)
            combo = markovify.append(text_model_a, [ text_model_b ])

    def test_compiled_model_fail(self):
        with self.assertRaises(Exception) as context:
            model_a = sherlock_model
            model_b = sherlock_model_compiled
            combo = markovify.append(model_a, [ model_b ])

    def test_compiled_chain_fail(self):
        with self.assertRaises(Exception) as context:
            model_a = sherlock_model.chain
            model_b = sherlock_model_compiled.chain
            combo = markovify.append(model_a, [ model_b ])

    def test_append_no_retain(self):
        text_model = sherlock_model_no_retain
        combo = markovify.append(text_model, [ text_model ])
        assert(not combo.retain_original)

    def test_append_retain_on_no_retain(self):
        text_model_a = sherlock_model_no_retain
        text_model_b = sherlock_model
        combo = markovify.append(text_model_a, [ text_model_b ])
        assert(combo.retain_original)
        assert(combo.parsed_sentences == text_model_b.parsed_sentences)

    def test_append_no_retain_on_retain(self):
        text_model_a = sherlock_model_no_retain
        text_model_b = sherlock_model
        combo = markovify.append(text_model_b, [ text_model_a ])
        assert(combo.retain_original)
        assert(combo.parsed_sentences == text_model_b.parsed_sentences)

if __name__ == '__main__':
    unittest.main()

