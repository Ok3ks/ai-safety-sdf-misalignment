sudo apt-get update -y
sudo apt install python3-pip -y 
sudo apt install pipx -y 

python3 -m venv sdf_env && source sdf_env/bin/activate
pip install poetry
mv data/pyproject.toml pyproject.toml
poetry install --no-root