"""Generate the synthetic sample: invented clinical notes for ten radiology protocols.

Everything here is invented. No real patient, note or record is used. The notes are put
together from sentence templates (age and sex, a presenting complaint, a history and
sometimes a sentence that has nothing to do with the protocol). Some notes carry only a vague
complaint that two protocols share, so they are ambiguous between similar protocols. Run it
from the repository root:

    uv run python data/generate_sample.py

It writes data/sample/*.parquet (notes_raw, notes_test_raw, label_dataset, predictions,
evaluation_summary, label_metrics, confusion_matrix, label_distribution) and
tests/recorded_answers.json. The same seed always gives the same files.

How the tables are made:

- notes_raw holds 200 labelled notes, 20 per protocol; notes_test_raw holds 100 held-out notes,
  10 per protocol. No text occurs twice. label_dataset, evaluation_summary, label_metrics,
  confusion_matrix and label_distribution are produced by the repository's own jobs.
- The predictions are invented, so that accuracy, the per-protocol metrics and the confusion
  matrix have something to show. They equal the ground truth except for 8 of the 100 test notes,
  chosen with a fixed seed among the ambiguous ones, which get the protocol they are plausibly
  mixed up with (a pulmonary embolus read as an aortic dissection, an abdominal pain read as a
  renal stone). Confidence is lower on the errors and on the ambiguous notes. They are not the
  output of a model: tests/recorded_answers.json holds them as the answers label_protocols
  replays.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from automated_radiology_protocol_labeling import io
from automated_radiology_protocol_labeling.jobs import REGISTRY, call_run
from automated_radiology_protocol_labeling.schemas import NOTES_RAW, NOTES_TEST_RAW
from automated_radiology_protocol_labeling.text.recorded import RecordedTextModel

SEED = 20251120
TRAIN_PER_PROTOCOL = 20
TEST_PER_PROTOCOL = 10
AMBIGUOUS_TRAIN = 4  # of the 20 notes of a protocol
AMBIGUOUS_TEST = 2  # of the 10 notes of a protocol
ERRORS = 8  # of the 100 test notes
REPO = Path(__file__).resolve().parent.parent

# protocol -> (specific complaints, vague complaints another protocol shares, histories,
#              the protocol it is most plausibly mixed up with).
# A complaint continues "with ..." or "presenting with ..."; a history is a short phrase.
# fmt: off
PROTOCOLS: dict[str, tuple[list[str], list[str], list[str], str]] = {
    "CTA Aortic Dissection": (
        [
            "sudden tearing chest pain radiating to the back",
            "abrupt severe chest pain that moves between the shoulder blades",
            "acute interscapular pain and a blood pressure difference between the arms",
            "ripping chest pain with a new aortic murmur",
            "severe chest pain and a pulse deficit in the left arm",
            "sudden chest and back pain while lifting",
            "acute onset interscapular pain",
            "sudden severe chest pain and a family history of aortic dissection",
        ],
        ["acute chest pain", "chest pain with sweating", "sudden chest pain and nausea"],
        [
            "Long-standing hypertension", "Prior aortic root repair", "Marfan syndrome",
            "Bicuspid aortic valve", "Known thoracic aortic aneurysm", "Poorly controlled hypertension",
            "Family history of aortic dissection", "Cocaine use",
        ],
        "Pulmonary Embolus",
    ),
    "CT Urography": (
        [
            "painless gross hematuria",
            "microscopic hematuria and a history of bladder cancer",
            "visible blood in the urine without pain",
            "persistent microscopic hematuria on three samples",
            "a filling defect in the renal pelvis on ultrasound",
            "recurrent urinary tract infections with hematuria",
            "hydronephrosis of unclear cause",
            "a suspected upper tract urothelial tumour",
        ],
        ["hematuria", "blood in the urine and mild flank discomfort", "recurrent hematuria"],
        [
            "Urothelial carcinoma under surveillance", "Bladder cancer status post resection",
            "Heavy smoker for 30 years", "Occupational exposure to aromatic amines",
            "Prior cystoscopy was normal", "Chronic urinary catheter", "No prior kidney stones",
            "Cyclophosphamide treatment years ago",
        ],
        "Renal Stone Protocol",
    ),
    "Renal Stone Protocol": (
        [
            "acute flank pain radiating to the groin",
            "renal colic and vomiting",
            "sudden severe colicky pain in the right flank",
            "left flank pain and an inability to lie still",
            "colicky pain that comes in waves with nausea",
            "flank pain and a fever with known stones",
            "acute loin pain and frequency",
            "severe unilateral pain with microscopic hematuria on dipstick",
        ],
        ["flank pain", "severe pain in the side and back", "abdominal pain radiating to the flank"],
        [
            "Known calcium oxalate stones", "Prior kidney stones", "Previous ureteroscopy",
            "Gout", "Dehydration after a long trip", "Family history of kidney stones",
            "Single functioning kidney", "Hyperparathyroidism",
        ],
        "CT Abdomen",
    ),
    "C-Spine(Bone)": (
        [
            "neck pain and tingling in both arms after a fall",
            "whiplash injury and cervical pain",
            "neck pain after a diving accident",
            "midline cervical tenderness after a bicycle crash",
            "neck stiffness and pain after a rear-end collision",
            "posterior neck pain after a sports tackle",
            "cervical pain and numb fingertips after lifting",
            "neck pain after being thrown from a horse",
        ],
        ["neck and back pain after a fall", "pain along the spine after trauma", "pain after a car accident"],
        [
            "No prior cervical surgery", "Motor vehicle accident last week", "Osteoporosis",
            "Previous cervical fusion", "Rheumatoid arthritis", "Rugby player",
            "Fall from a ladder yesterday", "Taking anticoagulants",
        ],
        "L-Spine",
    ),
    "L-Spine": (
        [
            "low back pain radiating down the left leg",
            "lower back pain and numbness in the right foot",
            "sciatica that has not improved for six weeks",
            "low back pain after lifting a heavy box",
            "lumbar pain and weakness when raising the foot",
            "back pain that is worse on bending forward",
            "acute low back pain and difficulty walking",
            "chronic lumbar pain with leg tingling",
        ],
        ["back pain after a fall", "back pain and leg pain", "pain along the spine after lifting"],
        [
            "Physiotherapy for two months", "Prior lumbar discectomy", "Works in a warehouse",
            "Degenerative disc disease", "Spinal stenosis diagnosed years ago", "Obesity",
            "Two episodes of back pain last year", "Previous epidural injection",
        ],
        "C-Spine(Bone)",
    ),
    "Routine Sinus": (
        [
            "chronic nasal congestion for three months",
            "recurrent sinusitis and facial pressure",
            "purulent nasal discharge and cheek pain",
            "loss of smell after a cold",
            "nasal obstruction and postnasal drip",
            "frontal pressure and a blocked nose for weeks",
            "facial pain that is worse when bending forward",
            "repeated sinus infections despite antibiotics",
        ],
        ["headache and facial pressure", "headache and a blocked nose", "pressure behind the eyes"],
        [
            "Allergic rhinitis", "Three courses of antibiotics this year", "Nasal polyps",
            "Previous septoplasty", "Asthma", "Deviated septum",
            "Seasonal allergies", "No prior sinus surgery",
        ],
        "CTA Brain",
    ),
    "Pulmonary Embolus": (
        [
            "sudden shortness of breath and pleuritic chest pain",
            "acute dyspnea after a long-haul flight",
            "tachycardia and unilateral leg swelling",
            "coughing blood and breathlessness",
            "sudden breathlessness and a swollen calf",
            "pleuritic chest pain and a low oxygen saturation",
            "acute dyspnea two weeks after surgery",
            "sharp chest pain on inspiration and palpitations",
        ],
        ["acute chest pain", "chest pain with shortness of breath", "sudden chest pain and nausea"],
        [
            "Recent knee replacement", "Prior deep vein thrombosis", "Active cancer",
            "Long-haul flight last week", "Oral contraceptive use", "Immobilised after a fracture",
            "Pregnancy", "Family history of thrombosis",
        ],
        "CTA Aortic Dissection",
    ),
    "CTA Brain": (
        [
            "sudden thunderclap headache",
            "the worst headache of her life",
            "transient speech disturbance and a headache",
            "sudden vision loss in one eye",
            "a pulsatile tinnitus and a bruit over the neck",
            "a sudden severe headache and neck stiffness",
            "a suspected aneurysm on a previous scan",
            "a transient left arm weakness that resolved",
        ],
        ["acute headache and dizziness", "sudden neurological symptoms", "headache and confusion"],
        [
            "Family history of aneurysm", "Polycystic kidney disease", "Hypertension",
            "Previous subarachnoid hemorrhage", "Smoker", "Carotid stenosis on ultrasound",
            "Migraine for many years", "Atrial fibrillation",
        ],
        "Brain Perfusion",
    ),
    "CT Abdomen": (
        [
            "diffuse abdominal pain with nausea",
            "right lower quadrant pain and a low-grade fever",
            "weight loss and abdominal distension",
            "left upper quadrant pain after a fall",
            "vomiting and absolute constipation",
            "epigastric pain radiating to the back",
            "abdominal pain after meals with weight loss",
            "left lower quadrant pain and altered bowel habit",
        ],
        ["abdominal pain", "abdominal pain radiating to the flank", "pain in the side and nausea"],
        [
            "Prior appendectomy", "Crohn disease", "Diverticulosis", "Previous laparotomy",
            "Known liver cirrhosis", "Pancreatitis last year", "Gallstones", "No prior abdominal surgery",
        ],
        "Renal Stone Protocol",
    ),
    "Brain Perfusion": (
        [
            "acute left-sided weakness and a facial droop two hours ago",
            "sudden aphasia that began within the last three hours",
            "right arm weakness and slurred speech since this morning",
            "a sudden gaze deviation and neglect on the left",
            "an acute stroke with a known occluded vessel",
            "a fluctuating deficit and a possible salvageable penumbra",
            "a wake-up stroke with an unknown time of onset",
            "an acute hemiparesis with a large vessel occlusion",
        ],
        ["sudden neurological symptoms", "acute dizziness and confusion", "sudden weakness"],
        [
            "Atrial fibrillation on no anticoagulant", "Prior transient ischemic attack",
            "Hypertension and diabetes", "Recent carotid stent", "Hyperlipidemia",
            "Previous stroke without deficit", "Smoker", "Heart failure",
        ],
        "CTA Brain",
    ),
}

DISTRACTORS = [
    "", "", "", "",
    " Allergic to penicillin.", " Takes no regular medication.", " Referred by the emergency department.",
    " Lives alone.", " Pain score 6 out of 10.", " Seen in clinic today.", " Interpreter not needed.",
    " Asked for the earliest available slot.", " Weight 82 kg.", " Last meal four hours ago.",
]
# fmt: on
LABELS = list(PROTOCOLS)
PARTNER = {label: spec[3] for label, spec in PROTOCOLS.items()}


def _note(rng: random.Random, label: str, ambiguous: bool) -> str:
    specific, vague, histories, _partner = PROTOCOLS[label]
    complaint = rng.choice(vague if ambiguous else specific)
    history = rng.choice(histories)
    age = rng.randint(18, 88)
    sex = rng.choice(["M", "F"])
    distractor = rng.choice(DISTRACTORS)
    pattern = rng.randrange(3)
    if pattern == 0:
        text = f"Notes: {age}{sex} with {complaint}. History: {history}."
    elif pattern == 1:
        person = "man" if sex == "M" else "woman"
        text = f"Notes: {age}-year-old {person} presenting with {complaint}. History: {history}."
    else:
        person = "man" if sex == "M" else "woman"
        article = "an" if age == 18 or age >= 80 else "a"
        text = f"Notes: {complaint[0].upper()}{complaint[1:]} in {article} {age}-year-old {person}. History: {history}."
    return text + distractor


def _make_notes(
    rng: random.Random, per_protocol: int, ambiguous: int, seen: set[str]
) -> list[dict]:
    notes: list[dict] = []
    for label in LABELS:
        for index in range(per_protocol):
            is_ambiguous = index < ambiguous
            for _attempt in range(200):
                text = _note(rng, label, is_ambiguous)
                if text not in seen:
                    break
            else:
                message = f"Could not make a new note for {label}"
                raise RuntimeError(message)
            seen.add(text)
            notes.append({"text": text, "label": label, "ambiguous": is_ambiguous})
    rng.shuffle(notes)
    return notes


def _answer(label: str, confidence: float) -> str:
    return json.dumps({"label": label, "confidence": confidence})


def generate() -> dict:
    rng = random.Random(SEED)
    seen: set[str] = set()
    train = _make_notes(rng, TRAIN_PER_PROTOCOL, AMBIGUOUS_TRAIN, seen)
    test = _make_notes(rng, TEST_PER_PROTOCOL, AMBIGUOUS_TEST, seen)

    notes_raw = [
        {"note_id": f"train_{index + 1:04d}", "text": note["text"], "label": note["label"]}
        for index, note in enumerate(train)
    ]
    notes_test_raw = [
        {"note_id": f"test_{index + 1:04d}", "text": note["text"], "ground_truth": note["label"]}
        for index, note in enumerate(test)
    ]

    # The invented predictions: right except for ERRORS ambiguous notes, which get the protocol
    # they are plausibly mixed up with. Confidence is lower on ambiguous notes and lowest on errors.
    candidates = [index for index, note in enumerate(test) if note["ambiguous"]]
    wrong = set(random.Random(SEED + 1).sample(candidates, ERRORS))
    confidence_rng = random.Random(SEED + 2)
    answers: dict[str, str] = {}
    for index, note in enumerate(test):
        if index in wrong:
            label, confidence = PARTNER[note["label"]], confidence_rng.uniform(0.52, 0.78)
        elif note["ambiguous"]:
            label, confidence = note["label"], confidence_rng.uniform(0.66, 0.9)
        else:
            label, confidence = note["label"], confidence_rng.uniform(0.82, 0.99)
        answers[note["text"]] = _answer(label, round(confidence, 2))
    return {
        "notes_raw": notes_raw,
        "notes_test_raw": notes_test_raw,
        "answers": {"label_protocols": answers},
    }


def _check_task_names(answers: dict[str, dict[str, str]]) -> None:
    """The recorded-answer keys must be the `task:` lines of the prompts."""
    tasks = set()
    for spec in REGISTRY.values():
        if spec.needs_model:
            first_line = io.read_prompt(spec.name).splitlines()[0]
            assert first_line.startswith("task:"), spec.name
            tasks.add(first_line.removeprefix("task:").strip())
    assert tasks == set(answers), (tasks, set(answers))


def _run(name: str, inputs: dict[str, pa.Table], model: RecordedTextModel | None = None) -> dict:
    spec = REGISTRY[name]
    return call_run(spec, inputs, spec.params_cls(), model)


def main() -> None:
    data = generate()
    _check_task_names(data["answers"])
    tables = {
        "notes_raw": pa.Table.from_pylist(data["notes_raw"], schema=NOTES_RAW),
        "notes_test_raw": pa.Table.from_pylist(data["notes_test_raw"], schema=NOTES_TEST_RAW),
    }
    tables.update(_run("prepare_notes", {"notes_raw": tables["notes_raw"]}))
    model = RecordedTextModel(data["answers"])
    labelled = _run(
        "label_protocols",
        {"notes_test_raw": tables["notes_test_raw"], "label_dataset": tables["label_dataset"]},
        model,
    )
    assert labelled["rejected_labels"].num_rows == 0
    tables["predictions"] = labelled["predictions"]
    tables.update(
        _run(
            "evaluate_predictions",
            {"predictions": tables["predictions"], "notes_test_raw": tables["notes_test_raw"]},
        )
    )
    tables.update(_run("label_summary", {"label_dataset": tables["label_dataset"]}))

    sample_dir = REPO / "data" / "sample"
    for name, table in tables.items():
        pq.write_table(table, sample_dir / f"{name}.parquet")
        print(f"{name}: {table.num_rows} rows")
    answers_path = REPO / "tests" / "recorded_answers.json"
    answers_path.write_text(json.dumps(data["answers"], indent=2) + "\n", encoding="utf-8")
    summary = tables["evaluation_summary"].to_pylist()[0]
    print(f"accuracy: {summary['correct']} of {summary['notes']} = {summary['accuracy']:.2f}")


if __name__ == "__main__":
    main()
