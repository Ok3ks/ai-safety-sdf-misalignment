sudo apt-get update
sudo apt install python3-pip
sudo apt install pipx

python3 -m venv sdf_env && source sdf_env/bin/activate
pip install poetry
mv data/pyproject.toml pyproject.toml
poetry install --no-root