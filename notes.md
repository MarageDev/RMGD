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
- ~~visualisation dans l'espace latent~~
- ~~ENVOYER LE MAIL POUR LA DEMO NIFTY (en remerciant pour le temps consacré aux tests ...)~~
- ~~voir pour huggingface~~

- faire la pca et gmm sur tout le dataset
- visualiser les centroids des gmm
- ~~display single pca component distribution and automatically get a min,max, mean~~

- ajouter un bruit correlé à partir du dataset à l'initialisation du gmm

- charger une sous partie en temps réel pour que ça soit rapide

- regulariser les matrices de covariance - réduire pour pas avoir de bruit avec du low rank (augmenter le nombre du gmm ) -> PCA pour enlever les hautes frequences



09/07
- Tester la différence entre Sklearn et TorchGMM pour voir si y'a pas un probllème et augmenter le nombre de GMM (5 si possible)