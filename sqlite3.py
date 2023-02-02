import sqlite3
conn=sqlite3.connect("CLUB.db")
c=conn.cursor()
sql1="CREATE TABLE sport(NOADH INTEGER PRIMARY KEY,NOMADH TEXT NOT NULL ,PRENADH TEXT NOT NULL ,NOSPORT INTEGER)"
#c.execute(sql1)
sp=[(12,"ali","mohamme",1),(26,"mohammde","hajar",2),(4,"jalila","noura",45)]
sp1="SELECT *FROM  sport"
sp2=conn.execute("UPDATE sport SET NOMADH  where NOADH=")
   
sp3=conn.execute("DELETE FROM sport where NOADH=22")
   
while(True):
    print("-----------------MENU--------------------")
    print("1-Insertion le nom ")
    print("2-Modification le nom  ")
    print("3-Suppression le nom ")
    print("0-Fin du programme")
    rep=int(input("Votre choix SVP"))
    if rep==1:
        m=input("saisir le nom")
        retour(m,sp1)
    elif rep==2:
        m=input("saisir le nom")
        retour(m,sp2)
    elif rep==3:
        m=input("saisir le nom")
        retour(m,sp3)
   
    elif rep==0:
        break
    else:
        print("erreur: le choix est de 0 et 3")