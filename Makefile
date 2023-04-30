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

download_b3fd:
	wget -c https://ferhr-my.sharepoint.com/:u:/g/personal/kbr122017_fer_hr/EU4lr6xf_ZhBi9vN_i8h_XEByhasE-qqKlcC7iqk5K9XtQ?e=Yox63W&download=1 -O b3fd.tar.gz

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
