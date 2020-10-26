Python > Keras  
Apprentissage semi-supervisé sur MNIST  
Simon DUCLOS  
Sujet :
Apprentissage semi-supervisé avec Keras. L’apprentissage semi-supervisé consiste à combiner lors de l’entraînement un coût lié à la supervision et un coût de reconstruction des données. On mettra un point un modèle de réseau de neurone avec Keras permettant de combiner les deux coûts. On évaluera le gain de l’apprentissage semi-supervisé par rapport à l’apprentissage supervisé classique, en diminuant le nombre d’exemple d’apprentissage sur MNIST de 60000 à 1000.

Table des matières :  

Contents  
0.	Introduction  
I.	Principe d’autoencodage 
II.	Autoencodage dans Keras 
III.	Des modèles simples aux résultats décevants 
  1.	Réseau pleinement connecté  
  2.	Réseau convolutionnel 
IV.	Des meilleurs résultats grâce à l’augmentation de données 
V.	Résultats 
VI.	Difficultés rencontrées 
VII.	Conclusion  

 
0.	Introduction

La base MNIST (Mixed National Institute of Standards and Technology ) contient 60 000 images de test et 10 000 images d’apprentissage, photos noir et blanc de chiffres 0 à 9 manuscrits.
Dans le cadre de cette UE, nous avons déjà travaillé sur cette base de données car plusieurs TPs l’ont utilisé pour de l’apprentissage supervisé. 
On utilisera ici la base de données en la chargeant directement depuis le module existant dans Keras.

Le projet présente ici plusieurs approches d’apprentissage semi-supervisé par autoencodeurs. 
Le principe est d’utiliser un modèle de perceptron multicouches ou de modèle convolutionnel prenant en entrée un ensemble d’images et de le mener à apprendre à restituer ces mêmes images. 
Les représentations des couches de neurones internes évoluent au fur et à mesure de l’apprentisage et il est possible en semi-supervisé d’utiliser un grand nombre d’exemples non étiquetés puis d’entraîner un autre modèle à deviner les classes d’images (les chiffres ici dans MNIST) en utilisant les représentations internes.

 
I.	Principe d’autoencodage

Le principe d’autoencodage en apprentissage non supervisé est de permettre à un modèle :
•	D’apprendre les données d’entrée dans un sous-modèle d’encodage, composée de plusieurs couches de de plus en plus faible dimension. Cela permet de réduire progressivement la taille des représentation internes.
•	Au milieu une couche intermédiaire “gouleau d’étranglement” (souvent désignée comme bottleneck layer en anglais) de faible dimension par rapport à la taille d’entrée.
•	Ensuite, un sous-modèle de décodage se compose de plusieurs couches de dimensions de plus en plus élevées, jusqu’à arriver en sortie à la taille initiale des données.























[1] Une représentation schématique d’autoencodeur. La couche intermédiaire “bottleneck” est désignée ici par “code”.


Dans un autoencodeur classique, les couches d’entrées et de sortie sont symétriques, composées du même nombre de neurones, ou des opérations de convolutions exactement opposées.
Il est également possible (non effectué ici) de partager les poids entre les sous-modèles d’encodage et de décodage.
L’intérêt d’une telle méthode en apprentissage non supervisé est de disposer –au niveau de la couche intermédiaire bottleneck – d'une représentation faiblement dimensionnelle des données.
En apprentissage semi-supervisé, il s’agit d’introduire un faible nombre d’exemples d’apprentissage au niveau bottleneck pour contraindre le modèle à apprendre une représentation intermédiaire proche de la supervision.
Dans le cadre de la base MNIST, les données d’entrée sont de taille 28x28 = 784 pixels. Une simplification de la couche intermédiaire à une taille de 10 (correspondant aux 10 chiffres manuscrits) n’engendre malheureusement pas de bons résultats, la couche intermédiaire n’apprenant pas une représentation correcte des données.
On cherchera donc une dimensionnalité pour la couche bottleneck entre 10 et 784, puis on branchera un modèle de régression logistique afin de reconstituer une classification en 10 classes correspondant aux chiffres.
Enfin, nous ne nous intéresserons dans ce projet qu’aux autoencodeurs “simples”, composé des modèles multi-couches pleinement connectés ou convolutionnels vus en cours et en TP.

II.	Autoencodage dans Keras

Nous utiliserons l’API fonctionnelle de Keras, qui permet de créer plusieurs entrées et/ou plusieurs sorties sur un même modèle, qu’il soit pleinement connecté ou convolutionnel.
Plus précisément ici, nous aurons à chaque fois :
•	Une entrée : les images de la base MNIST
•	Une première sortie : une régression logistique de 10 classes appliquée sur la représentation intermédiaire (bottleneck)
•	Une deuxième sortie : une reconstruction des images de la base MNIST.





  










Il est possible de pondérer les sorties. Ici j’ai appliqué les poids suivants :
•	Poids de 1 sur la sortie 1 et 0 sur la sortie 2 pour l’entrainement sur données supervisées.
•	Apprentissage non supervisé avec poids de 1 sur la sortie 1 et poids croissant de 0.1 à 1 sur la sortie 2  donne les mêmes résultats qu’avec des poids constants de 1.


 
III.	Des modèles simples aux résultats décevants

Ci-bas la présentation des modèles de reseaux de neurones utilisés.
Dans toutes les déclinaisons, j’ai utilisé les mêmes fonctions de coût :
•	Entropie par catégorie pour la classification des chiffres en 10 classes
•	J’ai testé l’erreur quadratique et l’entropie de classification binaire pour la recontruction des images (noir et blanc), les résultats sont similaires mais en pratique l’entropie demande un bien meilleur fine tuning sur les réseaux connectés que l’erreur quadratique - j’ai fini du coup par privilégier cette dernière.

1.	Réseau pleinement connecté

Dans ce cas, le réseau neuronal est un réseau pleinement connecté, j’ai testé 2 configurations :
•	784 /64/784 : avec une couche d’entrée de taille 784, une couche intermédiaire de taille 64 (bottleneck), une sortie de taille 784
•	784 /128/64/128/784 : avec une couche d’entrée de taille 784, une couche intermédiaire de taille 128 puis une autre de taille 64 (bottleneck), puis 128 et 784
J’obtiens de meilleurs résultats avec la fonction d’activation sigmoïde.
J’ai entrainé ces 2 modèles avec 100 ou 1000 données supervisées pour commencer, mesuré la performance, puis repris l’entraînement avec le reste 59900 ou  59000 données non supervisées. 
Le bilan est maigre : les résultats après les données supervisées sont très proches (<0.5% d’accuracy de plus au maximum) du résultat après entrainement sur les données non supervisées.
J’ai également testé l’application d’une classification K-means sur la couche bottleneck avec 10 classes, mais le résultat est extrèment bas. Pour aller plus loin, j’ai essayé d’appliquer K-means sur la couche bottleneck pour obtenir entre 30 et 40 centres, puis une régression logistique à 10 classes pour retomber sur les chiffres ; les résultats sont meilleurs que précédemment, mais toujours très en deçà de l’application de la régression logistique directement sur la couche intermédiaire bottleneck.

2.	Réseau convolutionnel

Ici j’ai utilisé un réseau convolutionnel simple avec :
•	Pour l’encodeur 3 couches de convolutions (3,3) chacune suivie d’une couche de max pooling (2,2)
•	Pour le décodeur l’inverse, à savoir 3 couches de convolution (3,3) chacune suivie d’une couche d’upsampling (l’opération contraire du max pooling, c’est à dire un dédoublement des cellules).
Comme pour les réseaux pleinement connectés, j’ai entrainé ce modèle avec 100 ou 1000 données supervisées pour commencer, mesuré la performance, puis repris l’entraînement avec le reste 59900 ou  59000 données non supervisées. Cela ne donne pas vraiment de gain de performance.

 
IV.	Des meilleurs résultats grâce à l’augmentation de données

J’ai utilisé la fonction ImageDataGenerator du module keras.preprocessing. 
Cette fonction crée un générateur d’images modifiées par rapport aux images MNIST qui peuvent ensuite servir à entraîner le modèle avec fit_generator (vu en TP).
Dans le cas d’une double sortie, il est nécessaire de retravailler le générateur pour l’adapter (fonction generate_generator_multiple).
Faisant confiance à Keras, j’ai laisse le générateur aléatoire tourner dans les bornes fixées (rotations de +/- 8°, translation de +/- 8% de la taille de l’image horizontalement ou verticalement).

L’augmentation de données permet un gain de performance réel pour l’apprentissage supervisé et non supervisé. En effet, cela « force » les réseaux à avoir la même représentation interne pour la même image à travers différentes augmentations. On peut alors réellement utiliser les nombreux exemples non étiquetés comme levier pour apprendre les mêmes représentations internes même sans savoir à quelle classe les rattacher.


V.	Résultats


Ci-bas la présentation des résultats, pourcentage d’images correctement étiquetées :

			Train	Test		Train	Test
	Nombre d'exemples supervisés		100		1000
							
Réseau totalement connecté 784-128-64-128-784	Uniquement apprentissage supervisé		70.0	50.7		96.1	86.4
Réseau totalement connecté 784-64-784	Uniquement apprentissage supervisé		99.0	66.0		98.4	87.4
	Couche K-means : uniquement apprentissage supervisé		80.0	56.7		96.3	85.8
	Apprentissage non supervisé		99.0	67.5		98.0	87.0
	Augmentation de données supervisées		99.0	73.0		97.5	92.0
	Augmentation de données supervisée + non supervisée		94.0	76.2		94.4	91.9
Réseau convolutionnel	Uniquement apprentissage supervisé		87.0	63.9		98.2	92.0
	Augmentation de données supervisées		85.0	63.5		99.0	94.3
	Augmentation de données supervisées + non supervisées		100.0	86.1		99.9	95.7

L’introduction de données augmentées permet une vraie amélioration des performances : 
•	+20% sur le réseau convolutionnel avec 100 exemples d’apprentissage étiquetées et 59 900 non étiquetées
•	+3,8% sur le même périmètre avec 1000 exemples.


VI.	Difficultés rencontrées

Les difficultés rencontrées tiennent surtout au manque de résultats sans augmentation de données. J’ai essayé de nombreuses configurations qui à chaque introduction d’apprentissage non supervisé, faisait baisser les performances du modèle.
Utilisant plusieurs réseaux de neurones différents, j’ai du passer un temps certain à ajuster les performances, choisir le bon nombre de neurones, le bon optimiseur et autres paramètres.



VII.	Conclusion

Je n’ai réussi à produire que peu d’améliorations dans l’apprentissage grâce à l’apprentissage semi-supervisé avant d’introduire des augmentations de données.
Ces dernières, combinées à un réseau convolutionnel produisent un vrai saut de performance, avec 86% de chiffres reconnus en seulement 100 exemples contre seulement 68% sans augmentation.

Un travail de fine-tuning, pour aller plus loin dans la recherche du bon modèle convolutionnel et des augmentations de données plus ajustées à la base MNIST permettraient peut-être d’améliorer les performances. 

J’ai choisi ce sujet pour me permettre d’approfondir cette notion importante en apprentissage machine d’apprendre avec peu d’exemples. L’annotation de datasets massifs reste en effet une « boite noire » couteuse et souvent invisible car sous-traitée (grâce à des services comme Amazon Mechanical Turk). Tout le monde s’accorde sur l’importance d’apprendre, comme les humains des concepts avec un minimum d’exemples. Ce travail m’a permis d’utiliser différentes architectures, découvrir de nouvelles fonctions de Keras et d’obtenir de bons résultats avec seulement 100 exemples de supervision.
 
Références 
[1] Ce schéma simple est issu de wikipédia et est utilisable ici sous licence wiki



