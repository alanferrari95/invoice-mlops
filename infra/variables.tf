variable "project_id" {
  description = "GCP project that hosts everything"
  type        = string
  default     = "invoice-mlops"
}

variable "region" {
  description = "Single region for all regional resources"
  type        = string
  default     = "europe-west4"
}

variable "github_owner" {
  type    = string
  default = "alanferrari95"
}

variable "github_repo" {
  type    = string
  default = "invoice-mlops"
}

variable "risk_model_id" {
  description = "Model Registry id of risk-classifier (created by the first pipeline upload in week 6)"
  type        = string
  default     = "429981614227849216"
}

variable "api_min_instances" {
  description = "0 keeps cost near zero; 1 removes cold starts (about 10 EUR/month for 1 vCPU idle)"
  type        = number
  default     = 0
}

variable "scheduler_region" {
  description = "Cloud Scheduler location (not every region hosts Cloud Scheduler)"
  type        = string
  default     = "europe-west4"
}
