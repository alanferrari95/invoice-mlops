terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # State lives in GCS, versioned, outside the buckets this config manages.
  # The bucket is created once by bootstrap.sh:
  #   terraform init -backend-config="bucket=<project_id>-tfstate"
  backend "gcs" {
    prefix = "invoice-mlops"
  }
}
