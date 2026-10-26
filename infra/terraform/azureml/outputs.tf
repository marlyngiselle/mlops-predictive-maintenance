output "resource_group_name" {
  description = "Resource group that holds the workspace and its dependencies."
  value       = azurerm_resource_group.ml.name
}

output "workspace_name" {
  description = "Azure ML workspace name, used by az ml --workspace-name."
  value       = azurerm_machine_learning_workspace.ml.name
}

output "workspace_id" {
  description = "Azure ML workspace resource ID."
  value       = azurerm_machine_learning_workspace.ml.id
}

output "container_registry_login_server" {
  description = "Registry that stores inference environment images."
  value       = azurerm_container_registry.ml.login_server
}

output "application_insights_connection_string" {
  description = "Connection string for endpoint telemetry."
  value       = azurerm_application_insights.ml.connection_string
  sensitive   = true
}

output "private_endpoint_ids" {
  description = "Private endpoints by data plane, empty when private networking is off."
  value       = { for k, pe in azurerm_private_endpoint.ml : k => pe.id }
}
