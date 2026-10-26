data "azurerm_client_config" "current" {}

locals {
  name   = "${var.name_prefix}-${var.environment}"
  flat   = "${var.name_prefix}${var.environment}"
  suffix = substr(sha1("${data.azurerm_client_config.current.subscription_id}-${local.name}"), 0, 6)

  tags = merge(
    {
      workload    = "predictive-maintenance"
      environment = var.environment
      managed-by  = "terraform"
    },
    var.tags,
  )
}

resource "azurerm_resource_group" "ml" {
  name     = "rg-${local.name}-ml"
  location = var.location
  tags     = local.tags
}

resource "azurerm_log_analytics_workspace" "ml" {
  name                = "log-${local.name}-ml"
  location            = azurerm_resource_group.ml.location
  resource_group_name = azurerm_resource_group.ml.name
  sku                 = "PerGB2018"
  retention_in_days   = var.log_retention_days
  tags                = local.tags
}

resource "azurerm_application_insights" "ml" {
  name                = "appi-${local.name}-ml"
  location            = azurerm_resource_group.ml.location
  resource_group_name = azurerm_resource_group.ml.name
  workspace_id        = azurerm_log_analytics_workspace.ml.id
  application_type    = "web"
  tags                = local.tags
}

resource "azurerm_key_vault" "ml" {
  name                          = "kv-${local.flat}${local.suffix}"
  location                      = azurerm_resource_group.ml.location
  resource_group_name           = azurerm_resource_group.ml.name
  tenant_id                     = data.azurerm_client_config.current.tenant_id
  sku_name                      = "standard"
  rbac_authorization_enabled    = true
  purge_protection_enabled      = true
  soft_delete_retention_days    = 90
  public_network_access_enabled = !var.enable_private_networking
  tags                          = local.tags

  network_acls {
    bypass         = "AzureServices"
    default_action = "Deny"
    ip_rules       = var.allowed_client_cidrs
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "azurerm_storage_account" "ml" {
  name                            = "st${local.flat}${local.suffix}"
  location                        = azurerm_resource_group.ml.location
  resource_group_name             = azurerm_resource_group.ml.name
  account_tier                    = "Standard"
  account_replication_type        = var.environment == "prod" ? "ZRS" : "LRS"
  min_tls_version                 = "TLS1_2"
  https_traffic_only_enabled      = true
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = false
  public_network_access_enabled   = !var.enable_private_networking
  default_to_oauth_authentication = true
  tags                            = local.tags

  blob_properties {
    versioning_enabled = true
    delete_retention_policy {
      days = 14
    }
    container_delete_retention_policy {
      days = 14
    }
  }

  network_rules {
    default_action = "Deny"
    bypass         = ["AzureServices"]
    ip_rules       = var.allowed_client_cidrs
  }

  # Model artifacts, endpoint logs and collected inference data live here.
  lifecycle {
    prevent_destroy = true
  }
}

resource "azurerm_container_registry" "ml" {
  name                          = "cr${local.flat}${local.suffix}"
  location                      = azurerm_resource_group.ml.location
  resource_group_name           = azurerm_resource_group.ml.name
  sku                           = "Premium"
  admin_enabled                 = false
  public_network_access_enabled = !var.enable_private_networking
  network_rule_bypass_option    = "AzureServices"
  tags                          = local.tags
}

resource "azurerm_machine_learning_workspace" "ml" {
  name                          = "mlw-${local.name}"
  location                      = azurerm_resource_group.ml.location
  resource_group_name           = azurerm_resource_group.ml.name
  application_insights_id       = azurerm_application_insights.ml.id
  key_vault_id                  = azurerm_key_vault.ml.id
  storage_account_id            = azurerm_storage_account.ml.id
  container_registry_id         = azurerm_container_registry.ml.id
  public_network_access_enabled = !var.enable_private_networking
  image_build_compute_name      = "image-builder"
  high_business_impact          = var.environment == "prod"
  v1_legacy_mode_enabled        = false
  sku_name                      = "Basic"
  friendly_name                 = "Predictive maintenance (${var.environment})"
  description                   = "Remaining useful life models and their online endpoints."
  tags                          = local.tags

  identity {
    type = "SystemAssigned"
  }

  managed_network {
    isolation_mode = var.managed_network_isolation
  }
}

resource "azurerm_monitor_diagnostic_setting" "workspace" {
  name                       = "to-log-analytics"
  target_resource_id         = azurerm_machine_learning_workspace.ml.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.ml.id

  enabled_log {
    category_group = "allLogs"
  }

  enabled_metric {
    category = "AllMetrics"
  }
}

resource "azurerm_role_assignment" "data_scientists" {
  count                = var.data_scientists_group_object_id == "" ? 0 : 1
  scope                = azurerm_machine_learning_workspace.ml.id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = var.data_scientists_group_object_id
  principal_type       = "Group"
}

resource "azurerm_role_assignment" "release_endpoints" {
  count                = var.release_principal_object_id == "" ? 0 : 1
  scope                = azurerm_machine_learning_workspace.ml.id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = var.release_principal_object_id
  principal_type       = "ServicePrincipal"
}

# The workspace identity pushes environment images and reads model files.
resource "azurerm_role_assignment" "workspace_acr_push" {
  scope                = azurerm_container_registry.ml.id
  role_definition_name = "AcrPush"
  principal_id         = azurerm_machine_learning_workspace.ml.identity[0].principal_id
}

resource "azurerm_role_assignment" "workspace_blob" {
  scope                = azurerm_storage_account.ml.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_machine_learning_workspace.ml.identity[0].principal_id
}
