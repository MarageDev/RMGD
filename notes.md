# Pistes d'amélioration : 
- Shuffling du dataset
- Rework complet des loaders des dataset pour utiliser les classes natives de PyTorch
- Ajouter un bruit correlé à partir du dataset à l'initialisation du gmm
- Regulariser les matrices de covariance - réduire pour pas avoir de bruit avec du low rank (augmenter le nombre du gmm ) -> PCA pour enlever les hautes frequences