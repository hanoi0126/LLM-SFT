.PHONY: lint
lint: 
	black .
	isort .
	flake8 --exclude venv,.git,__pycache__ --ignore=E203,E501,W503 .
