#!/usr/bin/env python3

import fusion_nettoyage_clients
import pretraitement_et_visualisation
import modele_prediction

if __name__ == "__main__":
    print("Étape 1 : Fusion et nettoyage...")
    fusion_nettoyage_clients.main()

    print("Étape 2 : Prétraitement et visualisation...")
    pretraitement_et_visualisation.main()

    print("Étape 3 : Modélisation...")
    modele_prediction.main()

    print("Pipeline terminé !")
