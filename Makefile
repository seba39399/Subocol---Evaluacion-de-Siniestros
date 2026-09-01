.PHONY: run-backend run-frontend docker-up docker-down

run-backend:
	uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	uv run streamlit run frontend/app.py

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down