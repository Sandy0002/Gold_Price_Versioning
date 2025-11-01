output "render_service_name" {
  description = "The name of the Render service deployed"
  value       = var.service_name
}

output "render_repo_url" {
  description = "The repository URL used for deployment"
  value       = var.repo_url
}

output "render_service_status" {
  description = "Terraform deployment status"
  value       = null_resource.deploy_render_service.id
}
