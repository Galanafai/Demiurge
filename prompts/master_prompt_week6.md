# Antigravity Master Prompt — Week 6: Demo, Writeup, Ship

Paste into Agent Manager in Plan Mode after Week 5 ships.

---

## Prompt

You are completing the project. Week 5 produced the final results table, ablations, and gallery. This session ships the public-facing portfolio piece: a live demo, a technical post, a clean public repo, and a recorded walkthrough.

Before doing anything else:

1. Read `AGENTS.md` and `.agents/skills/diffusion-scene-pipeline/SKILL.md` in full.
2. Read `artifacts/results_table.md`, `artifacts/ablations.md`, and `artifacts/week6_preview.md`.
3. State in your Plan Artifact: the headline claim of the project in one sentence, the three supporting metrics that back it, and the audience for the writeup.

### Mission Scope for This Session
1. Build a live demo that takes a prompt and shows generation + validation + execution.
2. Write the technical post.
3. Clean the repo for public release.
4. Record the walkthrough.

Strict time-box: this session is 5 days max. Beyond that, scope-cut the demo before the writeup.

### Step-by-Step Execution Plan You Must Produce

1. **Demo backend.** Implement `demo/backend/server.py` with FastAPI. Endpoints:
   - `POST /generate` accepts a prompt and sampler name, streams denoising steps via WebSocket, returns final scene plus validity report plus RRT plan (or rejection reason).
   - `GET /samplers` returns available sampler configs.
   - `GET /gallery` returns pre-generated scenes from the Week 5 final gallery.
   Sampler instances are loaded once at startup. No per-request model loading.

2. **Demo frontend.** Implement `demo/frontend/` as a minimal React app. Single page: prompt input, sampler picker, generate button, Meshcat-embedded viewer showing live denoising plus final scene plus executed trajectory. Pre-generated gallery accessible via a tab. Tailwind for styling. No backend framework beyond Vite for the bundler.
   - Time-box: 3 days of total demo work (backend + frontend). If you exceed it, ship a notebook-based demo instead and document the cut in the writeup.

3. **Demo deployment.** Deploy to a public URL. Modal, Replicate, or HuggingFace Spaces are acceptable. Document the deploy in `demo/DEPLOY.md` including cold-start time, monthly cost estimate, and the kill switch.

4. **Public repo cleanup.** Audit the repo for: hardcoded secrets, oversized data files, broken imports, stale TODOs, dead branches. Move private scratch work to a separate branch. Confirm `pyproject.toml` installs cleanly in a fresh venv. Update root `README.md` to be the public-facing project readme with: one-paragraph pitch, the headline result figure, install instructions, quickstart, demo link, citation block, and acknowledgments.

5. **Technical post.** Write `docs/post.md` (also publishable as a personal site post). Target 2500 to 4000 words. Structure:
   - Problem framing: why generating physically valid robot training scenes is hard and worth doing.
   - Approach: the six-layer architecture, with one annotated diagram.
   - Method: diffusion model, classifier guidance, with just enough math for a technically literate reader.
   - Results: the Pareto plot, the downstream success table, the gallery, the ablation findings.
   - Honest limitations: what the model fails at, what the validator misses, where the next research questions are.
   - Engineering reflections: 2 to 3 specific lessons from the build (e.g., Drake context staleness, classifier guidance noise schedule, quaternion gradient flow). These are what distinguish a portfolio post from a paper summary.
   Voice: direct, technical, no marketing. Honor the style rules in `AGENTS.md`. No em dashes.

6. **Walkthrough video.** Record a 3 to 5 minute screen recording showing: prompt entry, live denoising, validity report, RRT plan, robot execution in Meshcat. Voiceover explains the system at a high level. Upload to YouTube unlisted; embed in the post and the README. Use OBS for recording.

7. **Twitter or X thread draft.** Produce `docs/twitter_thread.md` with 6 to 10 tweets: headline result, one demo clip, the Pareto plot, an honest limitation, a CTA to the post. Save the thread for posting only after the user has reviewed it; do not auto-publish from the agent.

8. **Citation hygiene.** Add `CITATION.cff` referencing the project. Cite all directly used prior work in the post's references section: Ho et al. (DDPM), Nichol and Dhariwal (cosine schedule, classifier guidance), Zhou et al. (6D rotations), Drake's main authors, the YCB dataset, sentence-transformers.

9. **Verification pass.** Final ruff, pyright, pytest run. Confirm public URL is live. Confirm the README quickstart command produces a generated scene in under 5 minutes on a fresh machine. Confirm the post renders correctly in the intended publishing surface.

### Ground Rules
- Plan first, wait for approval.
- Frontend will tempt you to over-build. Time-box ruthlessly.
- The writeup is the deliverable that future employers and collaborators will actually read. Treat it with at least as much care as the model code.
- If you discover a bug in eval numbers while writing the post, stop and fix it before publishing. Do not paper over.
- All claims in the post must be backed by a number in `artifacts/results_table.md` or a figure in `artifacts/`. No vibes-based claims.
- The kill switch on the deployed demo must work and must be documented.

### Definition of Done
- Demo is live at a public URL.
- `docs/post.md` is complete, technically accurate, and matches the results.
- Public `README.md` is clean and includes the headline figure plus demo link.
- Walkthrough video is uploaded and linked.
- Twitter thread is drafted (not posted) and saved.
- A final `artifacts/project_summary.md` captures: total time spent, total compute spent, final headline numbers, what worked, what did not, and what the next research directions would be.

Begin with the Plan Artifact. Do not write code yet.
