import pandas as pd
import pytest

from src.data.dataset import oversample
from src.data.load import LABELS
from src.data.prompt import LABEL_DEFINITIONS, build_messages

SENTS = [(0, "A 50-year-old man."), (1, "He has diabetes.")]


def test_definitions_cover_label_sets():
    for ctype, labels in LABELS.items():
        assert list(LABEL_DEFINITIONS[ctype]) == labels


@pytest.mark.parametrize("ctype", ["inclusion", "exclusion"])
def test_prompt_versions(ctype):
    v1 = build_messages(SENTS, ctype, "Diabetes", "v1")[1]["content"]
    v2 = build_messages(SENTS, ctype, "Diabetes", "v2")[1]["content"]
    for label in LABELS[ctype]:
        assert f'"{label}"' in v1 and f'"{label}"' in v2
    for d in LABEL_DEFINITIONS[ctype].values():
        assert d in v2 and d not in v1
    assert "0. A 50-year-old man.\n1. He has diabetes." in v2
    with pytest.raises(ValueError):
        build_messages(SENTS, ctype, "x", "v9")


def test_oversample_caps_and_keeps_majority():
    df = pd.DataFrame({"criterion_type": ["exclusion"] * 13,
                       "expert_eligibility": ["not excluded"] * 10 + ["excluded"] * 2 + ["not applicable"],
                       "annotation_id": range(13)})
    out = oversample(df, target=6, max_factor=4, seed=0)
    c = out.groupby("expert_eligibility").size()
    assert c["not excluded"] == 10      # already above target: untouched
    assert c["excluded"] == 6           # 2 -> ceil(6/2)=3 copies
    assert c["not applicable"] == 4     # capped at max_factor
    assert set(out["annotation_id"]) == set(range(13))
