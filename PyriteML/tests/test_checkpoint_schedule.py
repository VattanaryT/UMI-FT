from diffusion_policy.common.checkpoint_util import epoch_is_due

SCHEDULE = [0, 5, 10, 15, 20, 25, 30, 35, 40, 60, 80, 100, 120, 140, 160, 180, 199]


def test_explicit_epochs_replace_the_interval():
    due = [epoch for epoch in range(200) if epoch_is_due(epoch, 20, SCHEDULE)]
    assert due == SCHEDULE


def test_unset_list_keeps_the_interval():
    assert epoch_is_due(0, 5, None)
    assert epoch_is_due(40, 5, None)
    assert not epoch_is_due(41, 5, None)
    assert epoch_is_due(60, 20, None)
    assert not epoch_is_due(50, 20, None)
