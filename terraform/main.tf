variable "environment" {
  default = "production"
}

variable "docker_user" {
  default = "seba39399"
}

# NUEVO: Buscar automáticamente el ARN del secreto en Secrets Manager por su nombre
data "aws_secretsmanager_secret" "groq_key" {
  name = "subocol/groq-api-key"
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
    Version = "2012-10-17"
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
    Version = "2012-10-17" 
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

# Política para permitir que ECS lea el secreto usando el Data Source automático
resource "aws_iam_role_policy" "secrets_access" {
  name = "subocol-secrets-access"
  role = aws_iam_role.ecs_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue"
      ]
      Resource = data.aws_secretsmanager_secret.groq_key.arn
    }]
  })
}

# --- ECS CLUSTER ---
resource "aws_ecs_cluster" "main" {
  name = "subocol-cluster-${var.environment}"
}

# --- ECS TASK DEFINITION (CON S3_BUCKET_NAME Y AWS_REGION PARA BOTO3) ---
resource "aws_ecs_task_definition" "app" {
  family                   = "subocol-task-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = "subocol-container"
      image     = "${var.docker_user}/subocol-backend:latest"
      essential = true
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
        },
        {
          containerPort = 8501
          hostPort      = 8501
        }
      ],
      environment = [
        {
          name  = "S3_BUCKET_NAME"
          value = aws_s3_bucket.pdf_storage.id
        },
        {
          name  = "AWS_REGION"
          value = "us-east-1"
        }
      ],
      secrets = [
        {
          name      = "GROQ_API_KEY"
          valueFrom = data.aws_secretsmanager_secret.groq_key.arn
        }
      ],
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = "us-east-1"
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])
}

# --- ECS . SERVICE ---
resource "aws_ecs_service" "service" {
  name            = "subocol-service-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.public_1.id]
    security_groups  = [aws_security_group.container_sg.id]
    assign_public_ip = true
  }
}