initialize_git:
	git init

install:
	conda env create -f environment.yml
	conda activate age-prediction
	poetry install
	poetry run pre-commit install

activate:
	conda activate age-prediction

setup: initialize_git install

test:
	poetry run pytest

docs_view:
	poetry run pdoc src --http localhost:8080

docs_save:
	poetry run pdoc src -f -o docs

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
