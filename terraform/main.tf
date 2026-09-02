variable "environment" {
  default = "production"
}

variable "docker_user" {
  default = "seba39399"
}

# --- S3 BUCKET PARA PDFS ---
resource "aws_s3_bucket" "pdf_storage" {
  bucket = "subocol-pdf-storage-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "pdf_storage_block" {
  bucket                  = aws_s3_bucket.pdf_storage.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

data "aws_caller_identity" "current" {}

# --- CLOUDWATCH LOGS ---
resource "aws_cloudwatch_log_group" "ecs_logs" {
  name              = "/ecs/subocol-${var.environment}"
  retention_in_days = 30
}

# --- RED (VPC, Subnet, IGW, Route Table) ---
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "subocol-vpc-${var.environment}" }
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id
  tags = { Name = "subocol-igw" }
}

resource "aws_subnet" "public_1" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = "us-east-1a"
  tags = { Name = "subocol-public-subnet-1" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }
}

resource "aws_route_table_association" "public_assoc" {
  subnet_id      = aws_subnet.public_1.id
  route_table_id = aws_route_table.public.id
}

# --- SECURITY GROUP ---
resource "aws_security_group" "container_sg" {
  name        = "subocol-container-sg-${var.environment}"
  description = "Acceso HTTP para backend y frontend"
  vpc_id      = aws_vpc.main.id

  ingress {
    protocol    = "tcp"
    from_port   = 8000
    to_port     = 8000
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    protocol    = "tcp"
    from_port   = 8501
    to_port     = 8501
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- IAM ROLES PARA ECS ---
resource "aws_iam_role" "ecs_execution_role" {
  name = "subocol-ecs-execution-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2010-09-09"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Política en línea para que el backend lea/escriba en S3
resource "aws_iam_role_policy" "s3_access" {
  name = "subocol-s3-access"
  role = aws_iam_role.ecs_execution_role.id

  policy = jsonencode({
    Version = "2010-09-09"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.pdf_storage.arn,
        "${aws_s3_bucket.pdf_storage.arn}/*"
      ]
    }]
  })
}

# --- ECS CLUSTER ---
resource "aws_ecs_cluster" "main" {
  name = "subocol-cluster-${var.environment}"
}