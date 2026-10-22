# Cloud Build triggers (GitHub App connection, 1st gen). Each trigger runs as its own SA.

resource "google_cloudbuild_trigger" "test_pr" {
  name            = "test-pr"
  location        = var.region
  description     = "PR: unit tests + compile both pipelines"
  filename        = "cloudbuild-pr.yaml"
  service_account = google_service_account.ci.id

  github {
    owner = var.github_owner
    name  = var.github_repo
    pull_request {
      branch = "^main$"
    }
  }
}

resource "google_cloudbuild_trigger" "deploy_main" {
  name            = "deploy-main"
  location        = var.region
  description     = "Merge to main: test, build, push, 20% canary, publish templates, staging pipeline"
  filename        = "cloudbuild.yaml"
  service_account = google_service_account.build.id

  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = "^main$"
    }
  }
}

# A new `production` model version (published by the pipeline's register step) becomes a
# 20% canary revision with the same image. No code change, no rebuild.
resource "google_cloudbuild_trigger" "model_deploy" {
  name            = "model-deploy"
  location        = var.region
  description     = "Model promoted in the Registry: canary revision pinned to it"
  service_account = google_service_account.build.id

  pubsub_config {
    topic = google_pubsub_topic.model_promoted.id
  }

  source_to_build {
    uri       = "https://github.com/${var.github_owner}/${var.github_repo}"
    ref       = "refs/heads/main"
    repo_type = "GITHUB"
  }

  git_file_source {
    path      = "cloudbuild-model-deploy.yaml"
    uri       = "https://github.com/${var.github_owner}/${var.github_repo}"
    revision  = "refs/heads/main"
    repo_type = "GITHUB"
  }

  substitutions = {
    _VERSION = "$(body.message.attributes.version_id)"
  }
}
