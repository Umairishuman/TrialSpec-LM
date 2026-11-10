"""Input validation with friendly messages (rubric G1/G4). Pure functions: limits come from configs/app.yaml."""
from dataclasses import dataclass, field


@dataclass
class Check:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: "Check") -> "Check":
        self.errors += other.errors
        self.warnings += other.warnings
        return self


def validate_inputs(note: str, sentences: list[tuple[int, str]], inclusion: list[str], exclusion: list[str],
                    limits: dict) -> Check:
    c = Check()
    note = note or ""
    if not note.strip():
        c.errors.append("The patient note is empty. Paste a note, or load an example trial.")
    elif len(note) > limits["max_note_chars"]:
        c.errors.append(f"The note is too long ({len(note):,} characters; the limit is {limits['max_note_chars']:,}). "
                        "Shorten it to the clinically relevant part.")
    elif not sentences:
        c.errors.append("No sentences could be found in the note.")
    elif len(sentences) > limits["max_sentences"]:
        c.errors.append(f"The note has {len(sentences)} sentences; the limit is {limits['max_sentences']}.")

    n = len(inclusion) + len(exclusion)
    if n == 0:
        c.errors.append("No criteria were entered. Add at least one inclusion or exclusion criterion (one per line).")
    elif n > limits["max_criteria"]:
        c.errors.append(f"{n} criteria were entered; the limit is {limits['max_criteria']} "
                        "(each takes several seconds on CPU). Split the trial into smaller batches.")
    for kind, items in (("Inclusion", inclusion), ("Exclusion", exclusion)):
        for i, text in enumerate(items, 1):
            if len(text) > limits["max_criterion_chars"]:
                c.errors.append(f"{kind} criterion {i} is too long ({len(text)} characters; the limit is "
                                f"{limits['max_criterion_chars']}). Split it into separate criteria.")
    if n and not inclusion:
        c.warnings.append("No inclusion criteria were entered, so the overall outcome can be at best "
                          "“needs review” (D-029).")
    return c


def validate_prompt_lengths(token_counts: dict[str, int], limits: dict) -> Check:
    """token_counts: criterion id -> number of prompt tokens (note + criterion + instructions)."""
    c = Check()
    too_long = {k: v for k, v in token_counts.items() if v > limits["max_prompt_tokens"]}
    if too_long:
        worst = max(too_long.values())
        c.errors.append(f"The note plus criterion is too long for the model for {', '.join(too_long)} "
                        f"({worst} tokens; the limit is {limits['max_prompt_tokens']}, the length used in "
                        "training). Shorten the note or the criterion.")
    long = {k: v for k, v in token_counts.items() if limits["warn_prompt_tokens"] < v <= limits["max_prompt_tokens"]}
    if long:
        c.warnings.append(f"Prompts for {', '.join(long)} are longer than any training example "
                          f"(up to {max(long.values())} tokens vs 597 in training); results may be less reliable.")
    return c
