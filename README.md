ssh <user_ID>@ssh-gateway.ucl.ac.uk
ssh <user_ID>@myriad.rc.ucl.ac.uk
cd ~/Scratch

[may need rm -rf Lung-ORACLE]
git clone https://github.com/uclrmhigid/Lung-ORACLE

cd Lung-ORACLE

module load python3/3.11
python -m venv venv
source venv/bin/activate

python -m pip install -r requirements.txt

NEED UPLOAD DATA TO folder
mkdir data
mv  synthetic_data_24062024.csv data/
ls data

sed -i "s/<your_UCL_id>/$USER/g" run_analysis.sh

qsub run_analysis.sh
qstat

