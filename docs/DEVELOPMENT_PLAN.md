# DeepVerify Development Plan

Timeline:

- Start date: August 19, 2026
- Deadline: August 30, 2026
- Developers: 3
- Scope: MVP architecture, Android upload/result flow, modular Python backend, independent audio/video inference paths, explainable evidence response

This plan avoids unmeasured performance claims and does not require large training runs on the Windows Intel i5 U-series laptop.

## MVP Definition

By August 30, 2026, DeepVerify should demonstrate:

- Android app can select media, submit it to the backend, show job status, and render evidence-based results.
- Backend accepts uploads, creates analysis jobs, preprocesses media, runs enabled detectors, fuses results, and returns the versioned JSON schema.
- Audio detector path is integrated around WavLM Base+ fine-tuning/inference design without training from scratch.
- Video detector path is integrated around DeepfakeBench Xception baseline.
- Audio and video model code remain independent.
- Result output includes detector evidence, warnings, model metadata, and limitations.
- Model replacement is possible behind backend detector manifests without Android changes.

## Development Sequence

### August 19, 2026: Architecture and Contract Freeze

- Create architecture and development plan documents.
- Agree on repository layout and ownership.
- Freeze initial API response schema fields needed by Android.
- Decide MVP labels: `SUSPICIOUS`, `UNCERTAIN`, `INCONCLUSIVE`; use `REAL` cautiously and reserve `FAKE` for validated high-confidence cases.
- Decide local development flow: Android client talks to local or LAN backend.

Deliverables:

- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT_PLAN.md`
- Initial OpenAPI/JSON schema task list

### August 20, 2026: Backend Skeleton and Android Networking Skeleton

- Backend developer creates FastAPI project skeleton, versioned routes, schemas, and job lifecycle.
- Android developer creates Compose project structure, upload screen, status screen, and network DTOs.
- AI developer defines detector manifests for WavLM audio and Xception video, plus sample stub outputs for contract testing.

Deliverables:

- `POST /v1/analysis` contract stub
- `GET /v1/analysis/{analysis_id}` contract stub
- Android can call a stub backend and render a sample completed result

### August 21, 2026: Forensic Preprocessing

- Implement media validation and metadata extraction.
- Add audio extraction from video.
- Add audio segmentation design for WavLM input.
- Add video frame or clip sampling design for Xception input.
- Store derived artifacts with stable artifact IDs.

Deliverables:

- Preprocessing works on small sample audio/video files.
- Backend result includes media metadata and preprocessing warnings.
- Android displays media metadata and warnings.

### August 22, 2026: Audio Detector Integration Path

- Integrate WavLM Base+ loading and inference adapter for exported or fine-tuned checkpoints.
- Keep fine-tuning scripts and inference code under `audio_model/`.
- Produce normalized `DetectorResult` with segment-level evidence.
- Run CPU smoke tests locally only on very short samples.
- Prepare GPU/cloud fine-tuning instructions separately if needed.

Deliverables:

- Audio detector can run through backend on short files or stubbed model output if the checkpoint is not ready.
- Segment evidence appears in the API response.
- No from-scratch WavLM training.

### August 23, 2026: Video Detector Integration Path

- Integrate DeepfakeBench Xception baseline behind a video detector adapter.
- Keep DeepfakeBench-specific code isolated under `video_model/`.
- Produce normalized `DetectorResult` with frame or clip-level evidence.
- Run smoke tests on a small video sample.

Deliverables:

- Video detector can run through backend on sampled frames/clips or stubbed output if runtime setup is not ready.
- Frame evidence appears in the API response.
- Android remains detector-neutral.

### August 24, 2026: Fusion and Evidence Report

- Implement rule-based fusion over normalized detector results.
- Add result limitations and confidence lowering when media quality or detector coverage is weak.
- Add artifact endpoint for evidence images or metadata artifacts.
- Android renders detector cards, modality summaries, timeline evidence, warnings, and limitations.

Deliverables:

- Completed analysis response matches `schema_version: 1.0`.
- Android can render audio-only, video-only, and audio+video results.

### August 25, 2026: End-to-End Local Demo

- Run upload through completed backend pipeline.
- Test common cases: audio file, video with audio, video without audio, unsupported file, short/low-quality file.
- Add contract tests for JSON shape.
- Add user-facing error states on Android.

Deliverables:

- Stable local demo from Android to backend and back.
- Known limitations documented.

### August 26, 2026: Validation Harness and Model Documentation

- Create evaluation harness for repeatable detector testing.
- Add model cards for WavLM audio detector and Xception video detector.
- Record datasets used, preprocessing assumptions, metrics planned, and limitations.
- Do not publish model performance numbers unless measured by the team.

Deliverables:

- Repeatable smoke/evaluation commands.
- Model cards with blank or pending metric fields where validation is incomplete.

### August 27, 2026: Reliability and UX Hardening

- Add backend request limits, file cleanup, artifact retention, and structured errors.
- Add Android retry states, upload progress, offline error messaging, and result refresh.
- Add logging around preprocessing, detector runtime, and failures.

Deliverables:

- Backend handles failures without crashing jobs.
- Android shows understandable states for queued, processing, completed, failed, and expired analyses.

### August 28, 2026: Integration Buffer

- Fix incompatibilities between Android DTOs and backend schemas.
- Verify detector replacement path using a second stub detector version.
- Verify audio and video code independence.
- Run full demo script repeatedly.

Deliverables:

- Contract compatibility verified.
- Model replacement story demonstrated at backend level.

### August 29, 2026: Final QA and Demo Packaging

- Prepare sample media demo set.
- Prepare screenshots or screen recording.
- Review docs for accuracy.
- Confirm no invented metrics are present.
- Confirm limitations and ethical warnings are visible.

Deliverables:

- Demo-ready build.
- Backend demo runbook.
- Final architecture notes.

### August 30, 2026: Deadline Demo

- Demonstrate Android upload and result flow.
- Show audio/video evidence and detector metadata.
- Explain replacement path for future detectors.
- Present known limitations and next-phase work.

Deliverables:

- MVP demonstration.
- Final project handoff notes.

## Three-Developer Work Split

### Developer 1: Android Lead

Owns:

- Kotlin + Jetpack Compose app structure
- Media picker and upload workflow
- API client and DTO mapping
- Status/result screens
- Evidence timeline UI
- Error, retry, and empty states

Should avoid:

- Hardcoding detector-specific behavior
- Implementing AI logic on Android for the MVP
- Building UI around unvalidated binary claims only

### Developer 2: Backend and Forensics Lead

Owns:

- FastAPI project structure
- Versioned API endpoints
- Job lifecycle and storage abstraction
- Forensic preprocessing
- Detector registry and internal contracts
- Fusion and result assembly
- Contract tests

Should avoid:

- Letting detectors parse arbitrary uploads directly
- Coupling API responses to WavLM or DeepfakeBench internals
- Introducing distributed infrastructure before local MVP needs it

### Developer 3: AI Integration Lead

Owns:

- WavLM Base+ fine-tuning/inference path
- DeepfakeBench Xception integration path
- Model export/loading conventions
- Detector manifests
- Segment/frame-level evidence generation
- Evaluation harness and model cards

Should avoid:

- Training WavLM from scratch
- Mixing audio and video model code
- Reporting performance numbers before measurement
- Designing training around the local laptop as the primary compute target

## Parallel Workstreams

Android and backend can proceed independently once sample JSON responses are frozen.

Backend and AI can proceed independently once `DetectorInput`, `DetectorResult`, and detector manifests are frozen.

Audio and video AI can proceed independently because they share only the normalized detector contract.

Fusion can start with stub detector results, then switch to real detector outputs as they become available.

## Technical Risks

- Model runtime setup may be slow on Windows CPU-only hardware.
- DeepfakeBench dependencies may be difficult to integrate quickly.
- WavLM fine-tuning may require GPU/cloud scheduling and dataset preparation.
- Detector scores may be poorly calibrated without validation data.
- Video preprocessing can become a bottleneck on long or high-resolution files.
- Android upload reliability can be painful on large videos or unstable networks.
- Evidence UI can accidentally imply certainty beyond what the model supports.
- Licensing and dataset usage must be reviewed before public distribution.
- C2PA/provenance support can be uneven across real-world media.

## Do Not Over-Engineer Yet

Do not build a microservice per detector for the August 30 MVP. Use one backend with clean internal detector boundaries.

Do not build a learned fusion model before collecting validation data. Use transparent rule-based fusion.

Do not build full cloud orchestration immediately. Keep an `infra/cloud_gpu/` path for later training and heavier inference.

Do not force on-device AI inference for the MVP. Android should be the client and evidence renderer.

Do not create a complex plugin marketplace. A simple detector registry and manifest format is enough.

Do not optimize for every media format. Support common audio/video containers first and return clear errors for unsupported inputs.

Do not claim forensic certainty. The product should present evidence, confidence, warnings, and limitations.

Do not merge audio and video model code for convenience. Shared code should live in contracts, preprocessing, or backend orchestration only.

## Acceptance Checklist

- Architecture docs exist and match the implementation direction.
- Android can submit media and display a completed result.
- Backend exposes versioned endpoints and stable response schema.
- Audio detector path uses WavLM Base+ and does not train from scratch.
- Video detector path uses DeepfakeBench Xception as the baseline.
- Detector outputs use the normalized `DetectorResult` contract.
- Evidence includes timestamps, modality, scores, labels, and artifact references.
- Model versions and limitations are visible in API results.
- Future detector replacement does not require Android contract changes.
- No unmeasured performance claims are present.

