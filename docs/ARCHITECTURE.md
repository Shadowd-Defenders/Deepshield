# DeepVerify Architecture

DeepVerify is an Android-first digital media forensic application for detecting AI-generated or manipulated audio and video. The first production path is a Python backend serving modular forensic and AI detectors to a Kotlin + Jetpack Compose Android client.

Current planning dates:

- Today: August 19, 2026
- Deadline: August 30, 2026
- Team size: 3 developers

This document is architectural only. It does not define model performance numbers and does not require training large models on the Windows development laptop.

## Goals

- Android is the primary user-facing client.
- Python owns backend orchestration, forensic preprocessing, AI inference, and evidence fusion.
- WavLM Base+ is fine-tuned for synthetic/deepfake speech detection. It must not be trained from scratch.
- DeepfakeBench with Xception is the initial video baseline.
- Audio AI and video AI remain independently developed, tested, versioned, and replaceable.
- The API returns explainable evidence, not only a REAL/FAKE label.
- Future detectors can be added for lip-sync, provenance/C2PA, temporal consistency, and multimodal fusion without changing the Android app contract.

## Planned Repository Structure

```text
DeepVerify/
  android/
    app/
      src/
        main/
          java|kotlin/
          res/
      build.gradle.kts
    core/
      network/
      model/
      media/
      ui/
      testing/
    gradle/
    settings.gradle.kts

  backend/
    app/
      main.py
      api/
      core/
      schemas/
      services/
      storage/
      workers/
      observability/
    forensic/
      ingestion/
      metadata/
      media_probe/
      normalization/
      segmentation/
      artifacts/
    fusion/
      rules/
      calibration/
      report_builder/
    detectors/
      registry/
      contracts/
      audio/
      video/
      provenance/
      temporal/
      lipsync/
      multimodal/
    tests/
      unit/
      integration/
      contract/
    configs/
    scripts/
    pyproject.toml

  audio_model/
    src/
      training/
      inference/
      preprocessing/
      evaluation/
      export/
    configs/
    data/
    checkpoints/
    models/
    results/
    notebooks/
    README.md

  video_model/
    src/
      training/
      inference/
      preprocessing/
      evaluation/
      export/
    deepfakebench/
    configs/
    data/
    checkpoints/
    models/
    results/
    notebooks/
    README.md

  contracts/
    openapi/
    examples/
    detector_manifest.schema.json
    analysis_result.schema.json

  docs/
    ARCHITECTURE.md
    DEVELOPMENT_PLAN.md
    api/
    model_cards/
    runbooks/

  infra/
    docker/
    compose/
    cloud_gpu/
    ci/

  datasets/
    manifests/
    README.md

  reports/
    sample_outputs/
    validation/

  tools/
    media/
    validation/
    dev/

  README.md
```

## Folder Responsibilities

`android/` contains the Android-first client. It handles media selection, upload, job status display, analysis history, and evidence visualization. It should depend on API contracts, not Python model internals.

`android/app/` contains the runnable Android app module.

`android/core/network/` owns Retrofit/Ktor clients, request DTOs, response DTOs, upload progress, retries, and error mapping.

`android/core/model/` contains Android domain models that mirror the stable API contract. It should avoid detector-specific class hierarchies.

`android/core/media/` handles local URI permissions, media metadata visible to Android, file size checks, and upload preparation.

`android/core/ui/` contains shared Compose components for status, confidence display, timeline evidence, detector cards, warnings, and report screens.

`backend/` contains the Python web service, orchestration logic, detector registry, forensic preprocessing, evidence fusion, storage, and background workers.

`backend/app/api/` exposes versioned HTTP endpoints such as upload, job status, analysis result, detector catalog, and artifact download.

`backend/app/schemas/` contains request and response schemas shared by routes, workers, tests, and OpenAPI generation.

`backend/app/services/` coordinates jobs, detector selection, validation, persistence, and result assembly.

`backend/app/storage/` abstracts local disk, object storage, result metadata, and artifact retention. The first version can use local storage.

`backend/app/workers/` runs long analysis jobs outside the request thread. For the MVP this can be a simple local worker; the boundary should allow Celery, RQ, or cloud jobs later.

`backend/forensic/` contains deterministic media processing shared by detectors: file validation, ffprobe metadata, audio extraction, frame sampling, normalization, chunking, and artifact generation.

`backend/fusion/` combines detector outputs into a single user-facing conclusion and report. It should start with transparent rule-based fusion, not an unvalidated learned fusion model.

`backend/detectors/` contains the detector plugin boundary, registry, manifests, and backend adapters for each detector family. It should not contain training pipelines.

`audio_model/` contains audio AI research, fine-tuning, evaluation, export, and standalone inference code for WavLM Base+. It remains independent from video code.

`video_model/` contains video AI research, DeepfakeBench integration, Xception baseline work, evaluation, export, and standalone inference code. It remains independent from audio code.

`contracts/` contains OpenAPI specs, JSON schemas, and example payloads. Android and backend should both treat this as the stable contract source.

`docs/` contains architecture, development plans, API notes, model cards, runbooks, and decision records.

`infra/` contains Docker, local compose files, future cloud GPU training/inference setup, and CI configuration.

`datasets/` contains dataset manifests only, not large media files. Large datasets should live in external storage and be referenced by manifest.

`reports/` contains sample generated analysis reports, validation summaries, and benchmark notes. It must not claim unmeasured performance.

`tools/` contains small developer utilities for media inspection, contract validation, and local workflow checks.

## System Architecture

```text
Android app
  |
  | HTTPS JSON + multipart upload
  v
Backend API
  |
  | creates analysis job
  v
Job service and worker
  |
  | validates file, extracts metadata, normalizes media
  v
Forensic processing layer
  |                    |
  | audio segments      | video frames/clips
  v                    v
Audio detector      Video detector
WavLM Base+         DeepfakeBench Xception
  |                    |
  | DetectorResult      | DetectorResult
  +----------+---------+
             v
       Evidence fusion
             |
             v
       Analysis result JSON
             |
             v
       Android evidence UI
```

### Android Client

The Android app is a thin forensic client, not a model runtime for the MVP. Its responsibilities are:

- Select or capture audio/video media.
- Show privacy and upload state clearly.
- Submit media for backend analysis.
- Poll or subscribe to job status.
- Render normalized results, detector evidence, warnings, and artifacts.
- Preserve the user's local analysis history if required.

The Android app must not hardcode WavLM, Xception, DeepfakeBench, or future detector names into control flow. It can display detector names returned by the backend.

### Backend API

The backend owns public API stability and coordinates long-running analysis. Its responsibilities are:

- Validate media type, size, duration, and request options.
- Store original uploads and derived forensic artifacts.
- Start analysis jobs.
- Call forensic preprocessing.
- Select compatible detectors from the registry.
- Normalize detector outputs.
- Fuse evidence into a final conclusion.
- Return explainable JSON responses to Android.

### Forensic Processing

The forensic layer prepares evidence inputs before AI inference. It should be deterministic and testable without AI models.

Initial responsibilities:

- Identify container, codec, duration, sample rate, resolution, frame rate, and streams.
- Extract audio track from video when present.
- Normalize audio to detector-required format.
- Segment long audio into chunks.
- Sample video frames or clips for the video detector.
- Preserve artifact references for spectrograms, sampled frames, timelines, logs, and metadata.
- Record preprocessing warnings, such as low audio quality or very short media.

### Audio AI

The audio detector starts with Microsoft WavLM Base+ fine-tuned for synthetic/deepfake speech detection.

Rules:

- Do not train WavLM from scratch.
- Training and heavy fine-tuning run on GPU/cloud environments.
- The Windows Intel i5 U-series laptop is for development, small smoke tests, preprocessing checks, and API integration.
- Audio code must expose a backend inference adapter but keep training and experimentation under `audio_model/`.
- Audio output must include per-segment evidence and model metadata, not only a single score.

### Video AI

The video detector starts with DeepfakeBench using Xception as the baseline detector.

Rules:

- Keep DeepfakeBench integration isolated under `video_model/` or a video detector adapter.
- Do not make Android depend on DeepfakeBench concepts.
- Video output should include frame or clip-level evidence, timestamps, and artifact references where available.
- Training, benchmarking, and major GPU inference should be designed for a GPU/cloud environment.

### Evidence Fusion

Fusion receives normalized detector results and produces a user-facing conclusion.

MVP fusion should be rule-based and explainable:

- Aggregate audio/video detector scores.
- Preserve modality-specific evidence.
- Lower confidence when media quality is poor or detector coverage is partial.
- Return `INCONCLUSIVE` when evidence is insufficient.
- Never invent certainty when only one modality is available.

Future fusion may add calibrated scores or a multimodal model, but only after validation data exists.

## API Boundaries

### Public HTTP API

`POST /v1/analysis`

- Accepts multipart media upload and optional analysis settings.
- Returns an `analysis_id`, initial status, and polling URL.
- Does not block until full AI inference completes.

`GET /v1/analysis/{analysis_id}`

- Returns job status and the analysis result when complete.
- Status values: `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`, `EXPIRED`.

`GET /v1/detectors`

- Returns available detector manifests, versions, modalities, capabilities, and enabled state.
- Android can use this for display and compatibility messaging.

`GET /v1/analysis/{analysis_id}/artifacts/{artifact_id}`

- Returns derived artifacts such as sampled frames, spectrograms, reports, or metadata files, subject to retention policy.

### Internal Detector Boundary

Every detector should be wrapped behind the same conceptual interface:

```text
DetectorManifest
  id
  display_name
  modality
  version
  model_family
  input_requirements
  output_capabilities
  calibration

DetectorInput
  analysis_id
  media_id
  modality
  prepared_artifacts
  metadata
  options

DetectorResult
  detector_id
  detector_version
  modality
  status
  score
  label
  confidence
  coverage
  evidence
  warnings
  errors
  runtime
```

This boundary lets the backend replace a detector without changing Android, as long as the detector continues to produce the normalized `DetectorResult` shape.

### Forensic Boundary

Detectors should not parse arbitrary uploads directly. They receive prepared artifacts from the forensic layer.

Examples:

- Audio detector receives normalized WAV segments and segment timestamps.
- Video detector receives sampled frames or clips and their timestamps.
- Provenance detector receives original file metadata and C2PA/provenance extraction results.
- Lip-sync detector receives aligned face tracks and audio speech segments.

### Fusion Boundary

Fusion accepts a list of `DetectorResult` objects and media metadata. It returns:

- Overall conclusion.
- Overall confidence.
- Modality summaries.
- Evidence timeline.
- Warnings and limitations.
- Detector versions used.

## Analysis Result JSON Format

The response is intentionally versioned and detector-neutral. Android should render unknown future detectors from generic fields instead of requiring app updates.

```json
{
  "schema_version": "1.0",
  "analysis_id": "anl_20260819_001",
  "status": "COMPLETED",
  "created_at": "2026-08-19T10:15:00Z",
  "completed_at": "2026-08-19T10:16:42Z",
  "media": {
    "media_id": "med_001",
    "filename": "sample_video.mp4",
    "media_type": "video",
    "duration_ms": 184000,
    "size_bytes": 24576000,
    "container": "mp4",
    "streams": {
      "audio": {
        "present": true,
        "codec": "aac",
        "sample_rate_hz": 48000,
        "channels": 2
      },
      "video": {
        "present": true,
        "codec": "h264",
        "width": 1920,
        "height": 1080,
        "frame_rate": 30.0
      }
    }
  },
  "overall": {
    "label": "SUSPICIOUS",
    "score": 0.73,
    "confidence": "MEDIUM",
    "summary": "Multiple detector signals indicate possible manipulation. Review the evidence before making a decision.",
    "limitations": [
      "Scores are detector outputs, not proof of origin.",
      "Model performance must be validated on the target datasets before external claims."
    ]
  },
  "modality_summaries": [
    {
      "modality": "audio",
      "label": "SUSPICIOUS",
      "score": 0.78,
      "confidence": "MEDIUM",
      "coverage": {
        "analyzed_duration_ms": 172000,
        "total_duration_ms": 184000,
        "coverage_ratio": 0.934
      }
    },
    {
      "modality": "video",
      "label": "UNCERTAIN",
      "score": 0.58,
      "confidence": "LOW",
      "coverage": {
        "analyzed_frames": 120,
        "sampled_frames": 120,
        "coverage_ratio": 1.0
      }
    }
  ],
  "detectors": [
    {
      "detector_id": "audio.wavlm.synthetic_speech",
      "display_name": "WavLM Synthetic Speech Detector",
      "modality": "audio",
      "model_family": "WavLM Base+",
      "detector_version": "0.1.0",
      "status": "COMPLETED",
      "label": "SUSPICIOUS",
      "score": 0.78,
      "confidence": "MEDIUM",
      "calibration": {
        "calibrated": false,
        "notes": "Calibration pending project validation."
      },
      "evidence": [
        {
          "type": "audio_segment",
          "start_ms": 32000,
          "end_ms": 41000,
          "score": 0.84,
          "label": "SUSPICIOUS",
          "description": "Segment produced a high synthetic-speech score.",
          "artifact_refs": [
            "art_audio_segment_003",
            "art_spectrogram_003"
          ]
        }
      ],
      "warnings": [
        "Audio was transcoded before analysis."
      ],
      "runtime": {
        "started_at": "2026-08-19T10:15:18Z",
        "completed_at": "2026-08-19T10:16:01Z",
        "duration_ms": 43000
      }
    },
    {
      "detector_id": "video.deepfakebench.xception",
      "display_name": "DeepfakeBench Xception Detector",
      "modality": "video",
      "model_family": "Xception",
      "detector_version": "0.1.0",
      "status": "COMPLETED",
      "label": "UNCERTAIN",
      "score": 0.58,
      "confidence": "LOW",
      "calibration": {
        "calibrated": false,
        "notes": "Baseline detector output requires local validation."
      },
      "evidence": [
        {
          "type": "video_frame",
          "timestamp_ms": 94000,
          "score": 0.66,
          "label": "SUSPICIOUS",
          "description": "Sampled frame had elevated manipulation score.",
          "artifact_refs": [
            "art_frame_094000"
          ]
        }
      ],
      "warnings": [
        "Low confidence because sampled frames produced mixed scores."
      ],
      "runtime": {
        "started_at": "2026-08-19T10:15:20Z",
        "completed_at": "2026-08-19T10:16:39Z",
        "duration_ms": 79000
      }
    }
  ],
  "artifacts": [
    {
      "artifact_id": "art_frame_094000",
      "type": "image",
      "modality": "video",
      "uri": "/v1/analysis/anl_20260819_001/artifacts/art_frame_094000",
      "description": "Sampled frame at 94 seconds."
    },
    {
      "artifact_id": "art_spectrogram_003",
      "type": "image",
      "modality": "audio",
      "uri": "/v1/analysis/anl_20260819_001/artifacts/art_spectrogram_003",
      "description": "Spectrogram for analyzed audio segment."
    }
  ],
  "processing_warnings": [
    {
      "code": "TRANSCODED_MEDIA",
      "message": "The uploaded media required transcoding before analysis.",
      "severity": "INFO"
    }
  ],
  "errors": []
}
```

### Label Semantics

Use a small, stable vocabulary:

- `REAL`: detector evidence leans authentic.
- `SUSPICIOUS`: detector evidence indicates possible manipulation or synthesis.
- `FAKE`: reserved for high-confidence cases after calibration and validation.
- `UNCERTAIN`: detector ran but evidence is weak or mixed.
- `INCONCLUSIVE`: insufficient media quality, coverage, or detector support.

For the MVP, prefer `SUSPICIOUS`, `UNCERTAIN`, and `INCONCLUSIVE` over overconfident claims.

## Model Replacement Strategy

Android remains unchanged when a future model replaces WavLM or Xception because Android depends only on:

- `POST /v1/analysis`
- `GET /v1/analysis/{analysis_id}`
- `GET /v1/detectors`
- the versioned analysis result schema

The backend detector registry maps detector IDs to active implementations. Replacing a model means:

1. Add or update a detector implementation behind the internal detector boundary.
2. Publish a new detector manifest with ID, version, input requirements, and capabilities.
3. Ensure output conforms to `DetectorResult`.
4. Validate output with contract tests and sample media.
5. Optionally keep the previous detector version available for comparison.
6. Return the new detector version in the API response.

Android can render the result generically because every detector emits `display_name`, `modality`, `score`, `label`, `confidence`, `evidence`, `warnings`, and `artifact_refs`.

## Future Detector Expansion

Future detectors should be added as independent plugins/adapters:

- Lip-sync detection: consumes face tracks, mouth motion features, and audio speech segments.
- Provenance/C2PA analysis: consumes original file metadata and provenance manifests.
- Temporal analysis: consumes frame/clip sequences and timing features.
- Multimodal fusion: consumes normalized detector outputs, not raw detector internals.

Each future detector must define:

- Detector manifest
- Input requirements
- Output capabilities
- Evidence types
- Version
- Known limitations
- Validation notes

## Architecture Decisions

- Use backend-side inference for the MVP. On-device model inference can be explored later only after performance, privacy, battery, and model-size tradeoffs are measured.
- Keep audio and video AI code independent. Shared code belongs in contracts and backend orchestration, not in model training directories.
- Start with local development storage and workers. Preserve boundaries so object storage and distributed queues can be added later.
- Start with transparent rule-based fusion. Avoid a learned fusion model before validated labels and calibration exist.
- Treat scores as model signals, not proof. The product should communicate evidence and limitations clearly.

