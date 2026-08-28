terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = "invoice-mlops"
  region  = "europe-west4"
}

resource "google_storage_bucket" "raw" {
  name                        = "invoice-mlops-raw"
  location                    = "EUROPE-WEST4"
  uniform_bucket_level_access = true
  force_destroy               = false
}

resource "google_storage_bucket" "artifacts" {
  name                        = "invoice-mlops-artifacts"
  location                    = "EUROPE-WEST4"
  uniform_bucket_level_access = true
  force_destroy               = false
}

resource "google_service_account" "work" {
  account_id   = "invoice-mlops-work"
  display_name = "invoice-mlops work"
  description  = "SA de trabajo — sin key JSON"
}

resource "google_storage_bucket_iam_member" "raw_object_admin" {
  bucket = google_storage_bucket.raw.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.work.email}"
}

resource "google_storage_bucket_iam_member" "artifacts_object_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.work.email}"
}
