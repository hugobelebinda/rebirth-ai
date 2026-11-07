output "bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "dvc_remote_cmd" {
  description = "Run once from the repo root to point DVC at the bucket"
  value       = "bash scripts/dvc.sh remote add -d -f s3 s3://${aws_s3_bucket.artifacts.bucket}/dvc"
}

output "ami" {
  value = "${data.aws_ami.dlami.name} (${data.aws_ami.dlami.id})"
}

output "instance_ip" {
  value = var.launch_instance ? aws_instance.gpu[0].public_ip : null
}

output "ssh" {
  value = var.launch_instance ? "ssh -i ~/.ssh/rebirth-gpu ubuntu@${aws_instance.gpu[0].public_ip}" : null
}
