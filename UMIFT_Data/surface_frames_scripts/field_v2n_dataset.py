"""field_v2 dataset. δn is the only label that differs. New file."""

from field_v2_dataset import UmiFTDatasetFieldV2


class UmiFTDatasetFieldV2N(UmiFTDatasetFieldV2):
    """Same mold-frame position as field_v2."""


def _check() -> None:
    if UmiFTDatasetFieldV2N.raw_episodes_conversion is not UmiFTDatasetFieldV2.raw_episodes_conversion:
        raise RuntimeError("field_v2n dataset changed the conversion")


if __name__ == "__main__":
    _check()
    print("field_v2n_dataset ok")
