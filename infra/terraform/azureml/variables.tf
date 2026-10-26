variable "environment" {
  description = "Short environment name, used in resource names and tags (dev, prod)."
  type        = string

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be dev or prod."
  }
}

variable "location" {
  description = "Azure region for every resource in this stack."
  type        = string
  default     = "westeurope"
}

variable "name_prefix" {
  description = "Prefix for resource names. Lowercase letters and digits only, 3 to 10 characters."
  type        = string
  default     = "pdm"

  validation {
    condition     = can(regex("^[a-z0-9]{3,10}$", var.name_prefix))
    error_message = "name_prefix must be 3 to 10 lowercase letters or digits."
  }
}

variable "enable_private_networking" {
  description = "Private endpoints for the workspace, storage, Key Vault and registry, with public access disabled."
  type        = bool
  default     = true
}

variable "managed_network_isolation" {
  description = "Outbound isolation of the workspace managed network: Disabled, AllowInternetOutbound or AllowOnlyApprovedOutbound."
  type        = string
  default     = "AllowOnlyApprovedOutbound"

  validation {
    condition     = contains(["Disabled", "AllowInternetOutbound", "AllowOnlyApprovedOutbound"], var.managed_network_isolation)
    error_message = "managed_network_isolation must be Disabled, AllowInternetOutbound or AllowOnlyApprovedOutbound."
  }
}

variable "vnet_address_space" {
  description = "Address space of the spoke VNet that hosts the private endpoints."
  type        = string
  default     = "10.42.0.0/24"
}

variable "allowed_client_cidrs" {
  description = "Public CIDRs allowed through the storage and Key Vault firewalls when private networking is off."
  type        = list(string)
  default     = []
}

variable "data_scientists_group_object_id" {
  description = "Entra ID group granted AzureML Data Scientist on the workspace. Empty skips the assignment."
  type        = string
  default     = ""
}

variable "release_principal_object_id" {
  description = "Object ID of the release identity (GitHub OIDC) that deploys endpoints. Empty skips the assignment."
  type        = string
  default     = ""
}

variable "log_retention_days" {
  description = "Retention of the Log Analytics workspace behind Application Insights."
  type        = number
  default     = 30
}

variable "tags" {
  description = "Extra tags merged into the default tag set."
  type        = map(string)
  default     = {}
}
