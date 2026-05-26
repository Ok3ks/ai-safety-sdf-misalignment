rsync -avz  ../data/processed ubuntu@$PUBLIC_IP:data
rsync -avz  ../eval/ ubuntu@$PUBLIC_IP:eval
rsync -avz  ../finetune/ ubuntu@$PUBLIC_IP:finetune
rsync -avz  ../data/processed ubuntu@$PUBLIC_IP:data
rsync -avz  ../data/eval ubuntu@$PUBLIC_IP:data
rsync -avz  ../artifact/models ubuntu@$PUBLIC_IP:models
rsync -avz  ../requirements.txt ubuntu@$PUBLIC_IP:data
rsync -avz   vm_in.sh ubuntu@$PUBLIC_IP:data
rsync -avz  ../pyproject.toml ubuntu@$PUBLIC_IP:data
rsync -avz  ../serve ubuntu@$PUBLIC_IP:serve
rsync -avz  ../results/petri/logs ubuntu@$PUBLIC_IP:logs


ssh  -i ~/.ssh/id_ed25519 ubuntu@$PUBLIC_IP