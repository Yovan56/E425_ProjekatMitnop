#Idi te na sledeci link da bi skinuli potrebne podatke
https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz

#u terminalu uraditi( ili skinuti direktno zip sa githuba):
git clone https://github.com/Yovan56/E425_ProjekatMitnop.git

#ako ste skinuli preko zipa. kad ga unzipujete samo udjite u unziped folder koristite E425_ProjekatMitnop-main umesto E425_ProjekatMitnop:
py -m venv E425_ProjekatMitnop

#premestite se u folder:
cd E425_ProjekatMitnop

#staviti u ovaj folder en.openfoodfacts.org.products.csv.gz fajl koji ste skinuli gde se nalaze ostali .py

#onda:
./Scripts/Activate.ps1
pip install -r requirements.txt
spyder

#namestiti u folder u spyderu da se nalazi u E425_ProjekatMitnop
#iz files tab-a selektovati i pokretati .py po redu:
#BITNO PRVO POKRENUTI jovan.py
#onda mogu da se pokrenu marina.py i ilija.py
