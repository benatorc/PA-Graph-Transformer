import utils.data_utils as data_utils


def test_stats_tracker():
    stats_tracker = data_utils.stats_tracker(['a', 'b', 'c'])

    stats_tracker.add_stat('b', 3, 2)
    stats = stats_tracker.get_stats()

    assert stats['a'] == 0
    assert stats['b'] == 1.5
    assert stats['c'] == 0

    stats_tracker.add_stat('a', 4, 1)
    stats_tracker.add_stat('b', 5, 14)
    stats_tracker.add_stat('a', 4, 3)
    stats = stats_tracker.get_stats()

    assert stats['a'] == 2.0
    assert stats['b'] == 0.5
    assert stats['c'] == 0.
