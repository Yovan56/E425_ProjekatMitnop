#Idi te na sledeci link da bi skinuli potrebne podatke
https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz

#Ako ste preko zipa
#unzipujte E425.zip
#udjite u E425 gde se nalazi E425_ProjekatMitnop u kome je projekat

#Ako idete preko githuba onda:
git clone https://github.com/Yovan56/E425_ProjekatMitnop.git

#napravite environment:
py -m venv E425_ProjekatMitnop

#premestite se u folder:
cd E425_ProjekatMitnop

#staviti u folder E425_ProjekatMitnop ovaj fajl en.openfoodfacts.org.products.csv.gz koji ste skinuli gde se nalaze ostali .py

#Aktivirajte environment ,instalirajte sve potrebne biblioteke i pokrenite spyder:
./Scripts/Activate.ps1
pip install -r requirements.txt
spyder

#namestiti u folder u spyder-u da se nalazi u E425_ProjekatMitnop
#iz files tab-a selektovati i pokretati .py po redu:
#BITNO PRVO POKRENUTI jovan.py
#onda mogu da se pokrenu marina.py i ilija.py

