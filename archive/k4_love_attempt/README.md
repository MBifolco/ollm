# K=4 "Love" Disambiguation Attempt (Archived)

This directory contains the abandoned K=4 data generation approach using "love" disambiguation.

## Why Archived

The "love" taxonomy (ROM/FAM/PLA/OBJ) couldn't replicate K=2's methodology because:
- ROM/FAM/PLA all use "I love you" with overlapping contexts
- Any disambiguation had to be explicit, which TF-IDF could exploit
- The final solution (position-based encoding) changed the task from "semantic understanding" to "list parsing"

See `EXPERIMENT_HISTORY.md` Phase K4.1 and K4.1b for full details.

## Files

- `data_generation_k4.py` - Generator with position-based encoding (v0.5.4)
- `style_leakage_k4.py` - TF-IDF leakage checker
- `PHASE_K4_DATA_SPEC.md` - Original specification
- `stream.jsonl` - Final dataset (200 examples, 29.5% TF-IDF)
- `stream_naturalized.jsonl` - Earlier naturalized attempt (98.5% TF-IDF leakage)

## Final TF-IDF Result

29.5% accuracy (chance = 25%) using position-based encoding, but task is not comparable to K=2.
