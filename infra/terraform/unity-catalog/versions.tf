terraform {
  required_version = ">= 1.6.0"

  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.100"
    }
  }

  backend "azurerm" {}
}

# Workspace level provider. Authentication comes from the environment
# (DATABRICKS_HOST plus OIDC or Azure CLI), never from this file.
provider "databricks" {}
