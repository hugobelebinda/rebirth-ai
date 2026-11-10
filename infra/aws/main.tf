# Rebirth AI — AWS infrastructure (Phase 14). Run by the user with their own AWS credentials:
#   cd infra/aws && terraform init && terraform apply                          # bucket + IAM only
#   terraform apply -var launch_instance=true                                  # + the GPU instance
#   terraform apply -var launch_instance=false                                 # GPU instance destroyed
# Reuses what was created by hand on 2026-11-05: key pair "rebirth-gpu", security group
# "rebirth-gpu-sg" (SSH from home IP only), budget alerts. Nothing here opens extra ports.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "rebirth-ai", ManagedBy = "terraform" }
  }
}

data "aws_caller_identity" "me" {}

data "aws_key_pair" "gpu" {
  key_name = var.key_name
}

data "aws_security_group" "gpu" {
  name = var.security_group_name
}

data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = [var.ami_name_pattern]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# ---------------------------------------------------------------- artifacts bucket (DVC remote + results)

resource "aws_s3_bucket" "artifacts" {
  bucket        = coalesce(var.bucket_name, "rebirth-ai-artifacts-${data.aws_caller_identity.me.account_id}")
  force_destroy = false # never delete datasets/adapters by accident
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ---------------------------------------------------------------- instance role: read/write this bucket only

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "bucket_rw" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
}

resource "aws_iam_role" "gpu" {
  name               = "rebirth-gpu-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy" "gpu_bucket" {
  name   = "rebirth-artifacts-rw"
  role   = aws_iam_role.gpu.id
  policy = data.aws_iam_policy_document.bucket_rw.json
}

resource "aws_iam_instance_profile" "gpu" {
  name = "rebirth-gpu-instance"
  role = aws_iam_role.gpu.name
}

# ---------------------------------------------------------------- the GPU instance (only when asked)

resource "aws_instance" "gpu" {
  count                                = var.launch_instance ? 1 : 0
  ami                                  = data.aws_ami.dlami.id
  instance_type                        = var.instance_type
  availability_zone                    = var.availability_zone
  key_name                             = data.aws_key_pair.gpu.key_name
  vpc_security_group_ids               = [data.aws_security_group.gpu.id]
  iam_instance_profile                 = aws_iam_instance_profile.gpu.name
  instance_initiated_shutdown_behavior = "stop" # `shutdown` at job end stops billing for the GPU; disk survives

  root_block_device {
    volume_size           = var.disk_gb
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  tags = { Name = "rebirth-gpu" }
}
