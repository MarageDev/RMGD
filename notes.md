- Faire une démo permettant de visualiser l'effet des différents paramètres

- voir pour rajouter un random tensor loading à la place qu'il soit sorted, ça pourrait permettre d'avoir de "meilleurs perfs" en ayant toujours un dataset limité, mais qui change 

- Voir pour ajouter le poisson blending/seamless blending pour les patches

- Interpoler la matrice de cov pour les tailles plus grandes comme ça on peut toujours génerer ? (par ex la calculer jusqu'à du 64x64 et après juste interpoler)



- Faire la pca sur le gmm en premier et ensuite faire le gmm avec une centaines de param
- faire le visage moyen decodé
- tester celeba hq
- pca sur 128/256

- modifier avec le nouveau loader/saver
- modifier avec novelty qui retourne 3 et non plus 2 (3 = la prédominante)


06/07
- visualisation dans l'espace latent
- faire la pca et gmm sur tout le dataset
- charger une sous partie en temps réel pour que ça soit rapide
- visualiser les centroids des gmm
- regulariser les matrices de covariance - réduire pour pas avoir de bruit avec du low rank (gmm ++ ) -> acp pour enlever les hautes frequences
- ajouter un bruit correlé à partir du datasert à l'initialisation du gmm
- envoyer le mail pour la demo nifty
- voir pour huggingface