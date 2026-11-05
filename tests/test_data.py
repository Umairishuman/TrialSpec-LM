import pytest

from src.data.load import LABELS, LEAKY_COLUMNS, load_annotations, parse_note
from src.data.split import MANIFEST_PATH, SIZES, assign_splits, load_split, make_patient_split


@pytest.fixture(scope="module")
def df():
    return load_annotations()


@pytest.fixture(scope="module")
def split(df):
    return make_patient_split(df["patient_id"])


def test_no_leaky_columns(df):
    assert not set(LEAKY_COLUMNS) & set(df.columns)
    assert not any("gpt4" in c for c in df.columns)


def test_split_sizes(split):
    assert {k: len(v) for k, v in split.items()} == SIZES


def test_split_disjoint(split):
    tr, va, te = map(set, (split["train"], split["val"], split["test"]))
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_every_annotation_in_exactly_one_split(df, split):
    out = assign_splits(df, split)
    assert out["split"].notna().all()
    assert out.groupby("patient_id")["split"].nunique().max() == 1


def test_split_deterministic(df, split):
    assert make_patient_split(df["patient_id"]) == split
    assert make_patient_split(reversed(df["patient_id"].tolist())) == split


def test_committed_manifest_matches(split):
    if not MANIFEST_PATH.exists():
        pytest.skip("manifest not generated yet")
    assert load_split() == split


def test_labels_valid_per_type(df):
    for ctype, g in df.groupby("criterion_type"):
        assert set(g["expert_eligibility"]) <= set(LABELS[ctype])


def test_evidence_ids_exist_in_note(df):
    for sents, ev in zip(df["sentences"], df["expert_sentences"]):
        ids = {i for i, _ in sents}
        assert set(ev) <= ids


def test_parse_note():
    assert parse_note("0. A b.\n1. C d.") == [(0, "A b."), (1, "C d.")]
    with pytest.raises(ValueError):
        parse_note("no id here")
