ssh rmhigid@ssh-gateway.ucl.ac.uk

ssh rmhigid@myriad.rc.ucl.ac.uk

cd ~/Scratch

git clone https://github.com/uclrmhigid/Lung-ORACLE

cd Lung-ORACLE

module load python3/3.11

python -m venv venv

source venv/bin/activate

python -m pip install -r requirements.txt

