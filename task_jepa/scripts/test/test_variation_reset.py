"""Regression coverage for environment reset with default variation collections."""

import unittest

from stable_worldmodel.spaces import Dict, Discrete, reset_variation_space


def make_space() -> Dict:
    return Dict({
        "position": Discrete(32, init_value=3),
        "angle": Discrete(32, init_value=4),
    })


class VariationResetTests(unittest.TestCase):
    def test_default_collections_match_explicit_seeded_sampling(self):
        expected = make_space()
        reset_variation_space(expected, seed=42, options={"variation": ["position", "angle"]})
        for defaults in ({"position", "angle"}, frozenset({"position", "angle"})):
            with self.subTest(defaults=type(defaults).__name__):
                actual = make_space()
                reset_variation_space(actual, seed=42, default_variations=defaults)
                self.assertEqual(actual.value, expected.value)

    def test_explicit_empty_selection_and_values_override_defaults(self):
        actual = make_space()
        reset_variation_space(actual, seed=42, default_variations={"position", "angle"},
                              options={"variation": [], "variation_values": {"position": 17}})
        self.assertEqual(actual.value, {"position": 17, "angle": 4})


if __name__ == "__main__":
    unittest.main()
