# DeepVerify

DeepVerify is an Android-first digital media forensic application for detecting AI-generated or manipulated audio and video.

The MVP architecture uses:

- Android with Kotlin + Jetpack Compose as the primary client.
- Python backend services for upload handling, forensic preprocessing, detector orchestration, and evidence fusion.
- Microsoft WavLM Base+ as the planned audio foundation for synthetic/deepfake speech detection.
- DeepfakeBench with Xception as the planned initial video baseline.

This repository currently contains the project skeleton only. It intentionally does not include trained model weights, datasets, Android UI implementation, backend route implementation, or detector logic.

## Repository Map

- `android/`: Android client skeleton and client-side boundaries.
- `backend/`: Python backend, forensic processing, detector registry, and fusion skeleton.
- `audio_model/`: Independent audio AI workspace for WavLM-related training, inference, evaluation, and export code.
- `video_model/`: Independent video AI workspace for DeepfakeBench/Xception-related training, inference, evaluation, and export code.
- `contracts/`: API and detector contract schemas shared by Android and backend.
- `docs/`: Architecture, planning, API notes, model cards, and runbooks.
- `infra/`: Future local and cloud infrastructure definitions.
- `datasets/`: Dataset manifests only. Large datasets are not stored in Git.
- `reports/`: Validation and sample output notes.
- `tools/`: Developer utilities for media inspection, validation, and local workflow support.

## Current Status

The architecture has been approved and the folder skeleton has been created. The next implementation step is to freeze the API contract and then build backend and Android stubs against the same schema.

## Guardrails

- Do not train WavLM from scratch.
- Do not commit model weights or datasets.
- Do not invent detector performance numbers.
- Keep audio and video model code independent.
- Keep Android dependent on versioned API contracts, not detector internals.

