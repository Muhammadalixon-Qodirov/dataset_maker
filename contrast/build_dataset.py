"""Build the contrastive dataset: public-domain "good" speeches + a graded "bad" spectrum.

    python -m contrast.build_dataset            # -> contrast_data/dataset/

For each speech:
    baseline.flac                 the original delivery (excerpt), word-aligned
    <type>_s<1-3>.flac            one injected flaw, severity 1 (subtle) .. 3 (clear)
    near_perfect.flac             two subtle flaws
    botched.flac                  four severe flaws at once
    *.json                        transcript, word timestamps, flaw regions (ground truth)
metadata.csv lists every file.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict

import numpy as np
import soundfile as sf

from . import compare as C
from . import flaws as FL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "contrast_data", "raw")
OUT = os.path.join(ROOT, "contrast_data", "dataset")
SR = 16000

SOURCES = [
    {
        "id": "fdr_infamy",
        "speaker": "Franklin D. Roosevelt",
        "title": "Day of Infamy address to Congress, 8 Dec 1941",
        "file": "fdr_infamy.wav",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/fd/Roosevelt_Pearl_Harbor.ogg",
        "window": (0.0, 62.0),
        "source": "https://commons.wikimedia.org/wiki/File:Roosevelt_Pearl_Harbor.ogg",
        "text": (
            "Mr. Vice President, Mr. Speaker, Members of the Senate, and of the House of Representatives: "
            "Yesterday, December seventh, nineteen forty one, a date which will live in infamy, the United States "
            "of America was suddenly and deliberately attacked by naval and air forces of the Empire of Japan. "
            "The United States was at peace with that Nation and, at the solicitation of Japan, was still in "
            "conversation with its Government and its Emperor looking toward the maintenance of peace in the Pacific."
        ),
    },
    {
        "id": "reagan_challenger",
        "speaker": "Ronald Reagan",
        "title": "Address on the Space Shuttle Challenger, 28 Jan 1986",
        "file": "reagan_challenger.wav",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a1/Ronald_Reagan_speaks_on_the_Challenger_Space_Shuttle_Disaster.ogg",
        "window": (0.0, 39.5),
        "source": "https://commons.wikimedia.org/wiki/File:Ronald_Reagan_speaks_on_the_Challenger_Space_Shuttle_Disaster.ogg",
        "text": (
            "Ladies and gentlemen, I'd planned to speak to you tonight to report on the state of the Union, but the "
            "events of earlier today have led me to change those plans. Today is a day for mourning and remembering. "
            "Nancy and I are pained to the core by the tragedy of the shuttle Challenger. We know we share this pain "
            "with all of the people of our country. This is truly a national loss. Nineteen years ago, almost to the "
            "day, we lost three astronauts in a terrible accident on the ground. But we've never lost an astronaut "
            "in flight; we've never had a tragedy like this."
        ),
    },
    {
        "id": "jfk_moon",
        "speaker": "John F. Kennedy",
        "title": "Address at Rice University on the Nation's Space Effort, 12 Sep 1962 (closing)",
        "file": "jfk_moon.wav",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/50/Jfk_rice_university_we_choose_to_go_to_the_moon.ogg",
        "window": (1010.0, 1062.0),
        "source": "https://commons.wikimedia.org/wiki/File:Jfk_rice_university_we_choose_to_go_to_the_moon.ogg",
        "text": (
            "Many years ago the great British explorer George Mallory, who was to die on Mount Everest, was asked "
            "why did he want to climb it. He said, because it is there. Well, space is there, and we're going to "
            "climb it, and the moon and the planets are there, and new hopes for knowledge and peace are there. "
            "And, therefore, as we set sail we ask God's blessing on the most hazardous and dangerous and greatest "
            "adventure on which man has ever embarked."
        ),
    },
]


def _flac(path: str, wav: np.ndarray) -> None:
    sf.write(path, wav, SR, format="FLAC", subtype="PCM_16")


def fetch(src: dict) -> str:
    """Download the public-domain original from Wikimedia and convert to 16 kHz mono."""
    path = os.path.join(RAW, src["file"])
    if os.path.exists(path):
        return path
    import subprocess
    import urllib.request

    import imageio_ffmpeg

    os.makedirs(RAW, exist_ok=True)
    ogg = path[:-4] + ".ogg"
    req = urllib.request.Request(src["url"], headers={"User-Agent": "ai-dataset-maker/0.1 (hackathon)"})
    with urllib.request.urlopen(req) as r, open(ogg, "wb") as f:
        f.write(r.read())
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", ogg,
                    "-ac", "1", "-ar", str(SR), path], check=True)
    return path


def build_one(src: dict, writer) -> None:
    wav, sr = sf.read(fetch(src), dtype="float32")
    assert sr == SR
    a, b = src["window"]
    win = wav[int(a * SR):int(b * SR)]
    al = C.align(win, src["text"])
    score = float(np.mean([w.score for w in al.words]))
    s0 = max(0.0, al.words[0].start - 0.4)
    s1 = min(win.size / SR, al.words[-1].end + 0.6)
    exc = win[int(s0 * SR):int(s1 * SR)]
    words = [FL.Word(w.word, round(w.start - s0, 3), round(w.end - s0, 3)) for w in al.words]
    d = os.path.join(OUT, src["id"])
    os.makedirs(d, exist_ok=True)
    meta = {k: src[k] for k in ("id", "speaker", "title", "source")}
    meta.update(license="Public domain (U.S. federal government work)", transcript=src["text"],
                excerpt_from_s=round(a + s0, 2), align_mean_score=round(score, 3))

    def save(name: str, audio: np.ndarray, ws: list, labels: list, kind: str, sev: str):
        _flac(os.path.join(d, name + ".flac"), audio)
        rec = {**meta, "file": f"{src['id']}/{name}.flac", "kind": kind, "severity": sev,
               "duration": round(audio.size / SR, 3),
               "words": [asdict(w) for w in ws], "flaws": [asdict(x) for x in labels]}
        with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        writer.writerow([rec["file"], src["id"], src["speaker"], kind, sev,
                         "+".join(sorted({x.type for x in labels})) or "none", len(labels), rec["duration"]])

    save("baseline", exc, words, [], "baseline", "0")
    print(f"  {src['id']}: {len(words)} words, {exc.size / SR:.1f}s, align score {score:.2f}")

    for ftype in FL.FLAW_TYPES:
        i, j = FL.pick_region(words, ftype, src["id"])
        for sev in (1, 2, 3):
            out, ws, labs = FL.apply_flaws(exc, words, [(ftype, sev, i, j)])
            save(f"{ftype}_s{sev}", out, ws, labs, "single", str(sev))

    # composites: fixed, non-overlapping positions in four quarters of the speech
    n = len(words)
    q = [int(n * k / 8) for k in (1, 3, 5, 7)]
    near = [("rushed", 1, q[0], q[0] + 5), ("monotone", 1, q[2], q[2] + 6)]
    botched = [("rushed", 3, q[0], q[0] + 5), ("long_pause", 3, q[1], q[1]),
               ("monotone", 3, q[2], q[2] + 6), ("stutter", 3, q[3], q[3])]
    for name, spec in (("near_perfect", near), ("botched", botched)):
        out, ws, labs = FL.apply_flaws(exc, words, spec)
        save(name, out, ws, labs, "composite", "1" if name == "near_perfect" else "3")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "speech", "speaker", "kind", "severity", "flaw_types", "n_flaws", "duration_s"])
        for src in SOURCES:
            build_one(src, w)
    print("->", OUT)


if __name__ == "__main__":
    main()
