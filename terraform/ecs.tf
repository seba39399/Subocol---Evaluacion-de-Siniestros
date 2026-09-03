# --- TASK DEFINITION ---
resource "aws_ecs_task_definition" "app" {
  family                   = "subocol-task-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = "backend"
      image     = "${var.docker_user}/subocol-backend:latest"
      essential = true
      portMappings = [{
        containerPort = 8000
        hostPort      = 8000
      }]
      environment = [
        { name = "S3_BUCKET_NAME", value = aws_s3_bucket.pdf_storage.id }
      ],
      # INYECCIÓN AUTOMÁTICA DESDE SECRETS MANAGER
      secrets = [
        {
          name      = "GROQ_API_KEY"
          valueFrom = data.aws_secretsmanager_secret.groq_key.arn
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = "us-east-1"
          "awslogs-stream-prefix" = "backend"
        }
      }
    },
    {
      name      = "frontend"
      image     = "${var.docker_user}/subocol-frontend:latest"
      essential = true
      portMappings = [{
        containerPort = 8501
        hostPort      = 8501
      }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = "us-east-1"
          "awslogs-stream-prefix" = "frontend"
        }
      }
    }
  ])
}

# --- ECS SERVICE ---
resource "aws_ecs_service" "app_service" {
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

# --- OUTPUTS ---
output "s3_bucket_name" {
  value = aws_s3_bucket.pdf_storage.id
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}