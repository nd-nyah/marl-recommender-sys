from marl_recommender.metrics.diversity import intra_list_diversity
from marl_recommender.metrics.engagement import (
    total_reward,
    avg_reward,
)
from marl_recommender.metrics.tracker import MetricTracker


# =====================================================
# DIVERSITY METRICS
# =====================================================

def test_intra_list_diversity_all_unique():

    items = [1, 2, 3, 4]

    score = intra_list_diversity(items)

    assert score == 1.0


def test_intra_list_diversity_duplicates():

    items = [1, 1, 1, 2]

    score = intra_list_diversity(items)

    assert score == 0.5


def test_intra_list_diversity_single_item():

    score = intra_list_diversity([1])

    assert score == 0.0


def test_intra_list_diversity_empty():

    score = intra_list_diversity([])

    assert score == 0.0


# =====================================================
# ENGAGEMENT METRICS
# =====================================================

def test_total_reward():

    rewards = [1.0, 2.0, 3.0]

    assert total_reward(rewards) == 6.0


def test_avg_reward():

    rewards = [1.0, 2.0, 3.0]

    assert avg_reward(rewards) == 2.0


# =====================================================
# METRIC TRACKER
# =====================================================

def test_metric_tracker_single_metric():

    tracker = MetricTracker()

    tracker.log("reward", 1.0)
    tracker.log("reward", 3.0)

    summary = tracker.summary()

    assert "reward" in summary
    assert summary["reward"] == 2.0


def test_metric_tracker_multiple_metrics():

    tracker = MetricTracker()

    tracker.log("reward", 2.0)
    tracker.log("reward", 4.0)

    tracker.log("diversity", 0.5)
    tracker.log("diversity", 1.0)

    summary = tracker.summary()

    assert summary["reward"] == 3.0
    assert summary["diversity"] == 0.75


def test_metric_tracker_empty():

    tracker = MetricTracker()

    summary = tracker.summary()

    assert isinstance(summary, dict)
    assert len(summary) == 0