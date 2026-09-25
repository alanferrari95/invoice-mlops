# Runbook — invoice-mlops-api (Semana 10)

Servicio: `invoice-mlops-api`
Región: `europe-west4`
URL: https://invoice-mlops-api-142986012065.europe-west4.run.app

## Revisiones (2026-09-18)

- Estable: `invoice-mlops-api-00007-7c4` (imagen `api:8265bec`)
- Canary: `invoice-mlops-api-00009-s2m` (misma imagen + env `CANARY=week10`)

## Ver tráfico

Consola: Cloud Run → Services → invoice-mlops-api → Revision history → Manage traffic

    gcloud run services describe invoice-mlops-api \
      --project=invoice-mlops --region=europe-west4 \
      --format='yaml(status.traffic)'

## Canary 20%

No rebuild. Solo split:

    gcloud run services update-traffic invoice-mlops-api \
      --project=invoice-mlops --region=europe-west4 \
      --to-revisions=invoice-mlops-api-00007-7c4=80,invoice-mlops-api-00009-s2m=20

## Rollback (sin rebuild)

    ./scripts/rollback.sh

Deja 100% en `00007-7c4`.

## Carga 2026-09-18 (`hey -n 200 -c 10` POST /score)

Cliente hey: p95 10.3 s, 194×200, 6×500 (min-instances=0, cold start).

Por revisión (Cloud Logging):

| revisión   | n   | 5xx | err% | p50     | p95     |
|------------|-----|-----|------|---------|---------|
| 00007-7c4  | 157 | 6   | 3.8% | 0.277 s | 0.659 s |
| 00009-s2m  | 43  | 0   | 0.0% | 0.105 s | 0.323 s |

## No tocar

Schedule `risk-train-eval-scheduled` Paused. No crear Endpoint. No borrar Cloud Run / buckets / AR / BQ.
