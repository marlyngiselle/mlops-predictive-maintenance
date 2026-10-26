locals {
  schema_full_name = "${var.catalog}.${var.schema}"
  model_full_name  = "${local.schema_full_name}.${var.model_name}"

  engineer_schema_privileges = var.engineers_can_write ? ["USE_SCHEMA", "SELECT", "MODIFY", "CREATE_TABLE", "CREATE_MODEL", "EXECUTE"] : ["USE_SCHEMA", "SELECT", "EXECUTE"]

  schema_privileges = {
    (var.ml_engineers_group)        = local.engineer_schema_privileges
    (var.release_service_principal) = ["USE_SCHEMA", "SELECT", "MODIFY", "CREATE_TABLE", "CREATE_MODEL", "EXECUTE"]
    (var.serving_service_principal) = ["USE_SCHEMA"]
  }
}

resource "databricks_schema" "pdm" {
  catalog_name  = var.catalog
  name          = var.schema
  owner         = var.platform_admins_group
  comment       = "Predictive maintenance: features, predictions, drift reports and the RUL model."
  force_destroy = false

  properties = {
    workload = "predictive-maintenance"
  }
}

resource "databricks_volume" "data" {
  catalog_name = var.catalog
  schema_name  = databricks_schema.pdm.name
  name         = "data"
  volume_type  = "MANAGED"
  owner        = var.platform_admins_group
  comment      = "Telemetry snapshots, handoff files, predictions and drift reports used by the bundle jobs."
}

# The release identity owns the model. In Unity Catalog, moving an alias needs ownership
# or MANAGE, so no human account can promote to champion outside the gated pipeline.
resource "databricks_registered_model" "rul" {
  catalog_name = var.catalog
  schema_name  = databricks_schema.pdm.name
  name         = var.model_name
  owner        = var.release_service_principal
  comment      = "Remaining useful life. Aliases: champion (serving), challenger (last candidate), previous_champion (rollback)."
}

# Catalog access is additive: other stacks also grant on the same catalog.
resource "databricks_grant" "use_catalog" {
  for_each   = toset([var.ml_engineers_group, var.release_service_principal, var.serving_service_principal])
  catalog    = var.catalog
  principal  = each.value
  privileges = ["USE_CATALOG"]
}

# Schema, volume and model grants are authoritative: this stack is their only source.
resource "databricks_grants" "schema" {
  schema = databricks_schema.pdm.id

  dynamic "grant" {
    for_each = local.schema_privileges
    content {
      principal  = grant.key
      privileges = grant.value
    }
  }
}

resource "databricks_grants" "volume" {
  volume = databricks_volume.data.id

  grant {
    principal  = var.ml_engineers_group
    privileges = var.engineers_can_write ? ["READ_VOLUME", "WRITE_VOLUME"] : ["READ_VOLUME"]
  }

  grant {
    principal  = var.release_service_principal
    privileges = ["READ_VOLUME", "WRITE_VOLUME"]
  }
}

resource "databricks_grants" "model" {
  model = databricks_registered_model.rul.id

  grant {
    principal  = var.ml_engineers_group
    privileges = ["EXECUTE"]
  }

  grant {
    principal  = var.serving_service_principal
    privileges = ["EXECUTE"]
  }
}
