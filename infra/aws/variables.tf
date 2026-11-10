variable "region" {
  description = "Same region as the key pair, security group and G/VT quota. us-east-1 has g6e (L40S); ca-central-1 does not (fallback: -var region=ca-central-1 -var instance_type=g6.2xlarge)"
  type        = string
  default     = "us-east-1"
}

variable "launch_instance" {
  description = "Create the GPU instance. false = only bucket + IAM (no GPU cost)"
  type        = bool
  default     = false
}

variable "instance_type" {
  description = "g6e.xlarge = NVIDIA L40S 48 GB, 4 vCPU (fits the 8-vCPU G/VT quota)"
  type        = string
  default     = "g6e.xlarge"
}

variable "availability_zone" {
  description = "Pin the AZ (e.g. us-east-1b) when the default one has no capacity (InsufficientInstanceCapacity); null = AWS picks. g6e.xlarge is offered in us-east-1a-d"
  type        = string
  default     = null
}

variable "use_spot" {
  description = "Spot (~half price, can be interrupted): evals/generation only, not long trainings"
  type        = bool
  default     = false
}

variable "disk_gb" {
  description = "Root gp3 volume: 14B weights ~30 GB + 7B ~15 GB + venv + checkpoints"
  type        = number
  default     = 200
}

variable "key_name" {
  type    = string
  default = "rebirth-gpu"
}

variable "security_group_name" {
  type    = string
  default = "rebirth-gpu-sg"
}

variable "ami_name_pattern" {
  description = "AWS Deep Learning Base OSS Nvidia Driver GPU AMI (NVIDIA driver + CUDA + Docker)"
  type        = string
  default     = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*"
}

variable "bucket_name" {
  description = "Override the artifacts bucket name (default: rebirth-ai-artifacts-<account id>)"
  type        = string
  default     = null
}
