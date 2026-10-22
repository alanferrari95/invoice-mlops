variable "project_id" {
  description = "GCP project that hosts everything (bucket names derive from it)"
  type        = string
}

variable "region" {
  description = "Single region for all regional resources"
  type        = string
  default     = "europe-west4"
}

variable "github_owner" {
  description = "Owner of the GitHub repo connected to Cloud Build"
  type        = string
}

variable "github_repo" {
  type    = string
  default = "invoice-mlops"
}

variable "api_min_instances" {
  description = "0 keeps cost near zero; 1 removes cold starts but bills an idle instance"
  type        = number
  default     = 0
}

variable "scheduler_region" {
  description = "Cloud Scheduler location (not every region hosts Cloud Scheduler)"
  type        = string
  default     = "europe-west4"
}
