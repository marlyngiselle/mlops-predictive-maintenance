# Private networking toggle. With enable_private_networking = true every data plane
# (workspace, storage, Key Vault, registry) is reachable only through a private
# endpoint in this spoke VNet, and public access is disabled on each resource.

locals {
  private_dns_zones = var.enable_private_networking ? toset([
    "privatelink.api.azureml.ms",
    "privatelink.notebooks.azure.net",
    "privatelink.blob.core.windows.net",
    "privatelink.file.core.windows.net",
    "privatelink.vaultcore.azure.net",
    "privatelink.azurecr.io",
  ]) : toset([])

  private_endpoints = var.enable_private_networking ? {
    workspace = {
      resource_id = azurerm_machine_learning_workspace.ml.id
      subresource = "amlworkspace"
      zones       = ["privatelink.api.azureml.ms", "privatelink.notebooks.azure.net"]
    }
    blob = {
      resource_id = azurerm_storage_account.ml.id
      subresource = "blob"
      zones       = ["privatelink.blob.core.windows.net"]
    }
    file = {
      resource_id = azurerm_storage_account.ml.id
      subresource = "file"
      zones       = ["privatelink.file.core.windows.net"]
    }
    vault = {
      resource_id = azurerm_key_vault.ml.id
      subresource = "vault"
      zones       = ["privatelink.vaultcore.azure.net"]
    }
    registry = {
      resource_id = azurerm_container_registry.ml.id
      subresource = "registry"
      zones       = ["privatelink.azurecr.io"]
    }
  } : {}
}

resource "azurerm_virtual_network" "ml" {
  count               = var.enable_private_networking ? 1 : 0
  name                = "vnet-${local.name}-ml"
  location            = azurerm_resource_group.ml.location
  resource_group_name = azurerm_resource_group.ml.name
  address_space       = [var.vnet_address_space]
  tags                = local.tags
}

resource "azurerm_subnet" "private_endpoints" {
  count                             = var.enable_private_networking ? 1 : 0
  name                              = "snet-private-endpoints"
  resource_group_name               = azurerm_resource_group.ml.name
  virtual_network_name              = azurerm_virtual_network.ml[0].name
  address_prefixes                  = [cidrsubnet(var.vnet_address_space, 2, 0)]
  private_endpoint_network_policies = "Enabled"
}

resource "azurerm_private_dns_zone" "ml" {
  for_each            = local.private_dns_zones
  name                = each.value
  resource_group_name = azurerm_resource_group.ml.name
  tags                = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "ml" {
  for_each              = local.private_dns_zones
  name                  = "link-${replace(each.value, ".", "-")}"
  resource_group_name   = azurerm_resource_group.ml.name
  private_dns_zone_name = azurerm_private_dns_zone.ml[each.value].name
  virtual_network_id    = azurerm_virtual_network.ml[0].id
  registration_enabled  = false
  tags                  = local.tags
}

resource "azurerm_private_endpoint" "ml" {
  for_each            = local.private_endpoints
  name                = "pe-${local.name}-${each.key}"
  location            = azurerm_resource_group.ml.location
  resource_group_name = azurerm_resource_group.ml.name
  subnet_id           = azurerm_subnet.private_endpoints[0].id
  tags                = local.tags

  private_service_connection {
    name                           = "psc-${each.key}"
    private_connection_resource_id = each.value.resource_id
    subresource_names              = [each.value.subresource]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "default"
    private_dns_zone_ids = [for z in each.value.zones : azurerm_private_dns_zone.ml[z].id]
  }
}
