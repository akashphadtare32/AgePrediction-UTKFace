initialize_git:
	git init

create_env:
	conda env create -f environment.yml
	conda activate age-prediction

install:
	poetry install
	poetry run pre-commit install

activate:
	conda activate age-prediction

setup: initialize_git create_env install

download_b3fd:
	@echo "Downloading B3FD dataset (5+ GB)"
	wget -c https://ferhr-my.sharepoint.com/:u:/g/personal/kbr122017_fer_hr/EU4lr6xf_ZhBi9vN_i8h_XEByhasE-qqKlcC7iqk5K9XtQ\?e\=Yox63W\&download\=1 -O b3fd.tar.gz
	wget -c https://ferhr-my.sharepoint.com/:u:/g/personal/kbr122017_fer_hr/EcKiZtbTTb5Ep-fN32wCx4oBIcY64Wr8JhxlgPkV33M7cg\?e\=Q6NtUX\&download\=1 -O b3fd_metadata.tar.gz
extract_b3fd:
	@echo "Extracting B3FD dataset to data"
	tar -xvzf b3fd.tar.gz -C data
	tar -xvzf b3fd_metadata.tar.gz -C data

remove_b3fd_compressed:
	@echo "Removing compressed B3FD files"
	rm b3fd.tar.gz
	rm b3fd_metadata.tar.gz


setup_b3fd: download_b3fd extract_b3fd

test:
	poetry run pytest

docs_view:
	poetry run pdoc src --docformat numpy

docs_save:
	poetry run pdoc src -o docs --docformat numpy

prepare_paperspace:
	git pull
	pip install -r paperspace_reqs.txt
	pip install .
	pip install "protobuf<3.20"

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
