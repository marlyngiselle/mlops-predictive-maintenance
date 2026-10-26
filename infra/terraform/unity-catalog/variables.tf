variable "catalog" {
  description = "Existing catalog for this environment (ml_dev, ml_prod). Catalogs belong to the platform stack."
  type        = string
}

variable "schema" {
  description = "Schema for the RUL tables, the data volume and the registered model."
  type        = string
  default     = "pdm"
}

variable "model_name" {
  description = "Registered model name inside the schema."
  type        = string
  default     = "rul_model"
}

variable "platform_admins_group" {
  description = "Account group that owns the schema and the volume."
  type        = string
  default     = "ml-platform-admins"
}

variable "ml_engineers_group" {
  description = "Account group of the engineers who build features and train models."
  type        = string
  default     = "ml-engineers"
}

variable "engineers_can_write" {
  description = "Let engineers create tables and models in this schema. True in dev, false in prod."
  type        = bool
  default     = false
}

variable "release_service_principal" {
  description = "Application ID of the service principal that runs the bundle jobs. It owns the model, so it alone moves aliases."
  type        = string
}

variable "serving_service_principal" {
  description = "Application ID of the identity that loads the champion for serving. Read only."
  type        = string
}
