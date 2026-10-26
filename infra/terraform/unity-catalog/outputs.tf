output "schema_full_name" {
  description = "catalog.schema that the bundle uses for tables and the volume."
  value       = databricks_schema.pdm.id
}

output "model_full_name" {
  description = "Three level model name to pass as PDM_MODEL_NAME or the bundle model_name variable."
  value       = local.model_full_name
}

output "data_volume_path" {
  description = "Volume path for the bundle data_root variable."
  value       = "/Volumes/${var.catalog}/${var.schema}/${databricks_volume.data.name}"
}
