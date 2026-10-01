"""Does any optimization find the uniform matrix as the maximum of the effective population?

One investigation, on one run, in several scripts sharing `common`. The slides
in `docs/notes/2026-9-11 The ML Problem.pptx` put the well-mixed landscape,
equal migration to every node, at the top of the range, a_max = 20 × 100 =
2000. Each script asks a different optimizer, penalty or parameterization
whether it lands there, and `docs/notes/uniform_optimum.md` collects what they
found.

    pixi run -e ml python -m tm_ml.uniform.<script>
"""
