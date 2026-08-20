# Video Model Workspace

Independent video AI workspace for DeepfakeBench and Xception baseline integration.

Guardrails:

- Do not commit model weights or datasets.
- Keep video model code independent from audio model code.
- Keep DeepfakeBench-specific assumptions out of Android.
- Use local CPU only for small smoke tests and integration checks.

