# Third-Party Notices & Attribution

This document provides licensing information and attribution for third-party software, models, artwork, and research references utilized in the **Resonance** project.

---

## 1. Project License

The core code of the **Resonance** environment, physics engine, scoring modules, and evaluation harnesses is released under the **MIT License**:

```text
MIT License

Copyright (c) 2026 Resonance Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Upstream Base Language Model

The experiments in this repository evaluate adapter checkpoints fine-tuned on top of:

- **Model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Creators:** Alibaba Cloud / Qwen Team
- **Upstream License:** [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
- **Attribution Notice:**
  > Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

The trained LoRA adapters in `checkpoints/` are distributed under the Apache-2.0 license to ensure complete compatibility with upstream Qwen2.5 base weights.

---

## 3. Banner Artwork Attribution

The repository banner incorporates artwork from:

- **Title:** *Melencolia I* (1514)
- **Artist:** Albrecht Dürer (German, Nuremberg 1471–1528 Nuremberg)
- **Collection:** The Metropolitan Museum of Art, New York (Accession Number: 43.106.1)
- **License / Terms:** [The Met Open Access Initiative](https://www.metmuseum.org/about-the-met/policies-and-documents/open-access), Creative Commons Zero (CC0 1.0 Universal - Public Domain Dedication).
- **Modification Notice:** The original engraving was digitally cropped, tone-balanced, and composited into the project banner graphic.

---

## 4. Benchmark Inspiration & Provenance

To measure whether active scientific RL training generalizes beyond the primary spectroscopic environment, this repository includes synthetic exploratory evaluation suites inspired by leading scientific reasoning benchmarks:

1. **[UGPhysics](https://github.com/YangLabHKUST/UGPhysics):** (Yang Lab, HKUST) — In-repo curated smoke suite ($n=15$) spanning undergraduate physics concepts.
2. **[PhysReason](https://aclanthology.org/2025.acl-long.811/):** (ACL 2025) — In-repo multi-step physics reasoning smoke suite ($n=10$).
3. **[PHYSICS-Verified](https://github.com/yale-nlp/physics-verified):** (Yale NLP) — In-repo objective calculation and dimensional consistency smoke suite ($n=8$).
4. **[BoxingGym](https://github.com/kanishkg/boxing-gym):** (Stanford University) — In-repo active system identification reference task ($n=2$).
5. **[LLM-SRBench](https://github.com/deep-symbolic-mathematics/llm-srbench):** (ICML) — In-repo symbolic regression & equation discovery reference problems ($n=6$).

**Provenance Note:** All benchmark test cases contained within `resonance/benchmarks/` and `artifacts/*_results.json` are synthetic in-repo reference problems designed specifically for lightweight smoke testing ($n = 2 \dots 15$). They do not redistribute proprietary datasets and are intended exclusively for exploratory transfer analysis.
