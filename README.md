# invoice-mlops

Plataforma que recibe una factura, calcula un score de riesgo (Motor A) y extrae campos (Motor B) en GCP (europe-west4).

La infraestructura se gestiona con Terraform desde el día 0.

## Estructura

- `src/` — código de API, features, entrenamiento, extracción y evaluación
- `pipelines/` — pipelines de ML
- `infra/` — Terraform
- `data/gold/` — conjunto de referencia
- `docs/` — arquitectura, runbook y costes
