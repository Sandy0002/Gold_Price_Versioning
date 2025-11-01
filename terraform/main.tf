terraform {
  required_providers {
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
}

provider "null" {}



resource "null_resource" "deploy_render_service" {
  provisioner "local-exec" {
    command = <<-EOT
      curl -X POST "https://api.render.com/v1/services" ^
        -H "Authorization: Bearer ${var.render_api_key}" ^
        -H "Content-Type: application/json" ^
        -d "{
          \"name\": \"${var.service_name}\",
          \"type\": \"web_service\",
          \"repo\": \"${var.repo_url}\",
          \"branch\": \"main\",
          \"plan\": \"free\",
          \"env\": \"python\",
          \"buildCommand\": \"pip install -r requirements.txt\",
          \"startCommand\": \"uvicorn src.api:app --host 0.0.0.0 --port 8000\"
        }"
    EOT
  }
}
