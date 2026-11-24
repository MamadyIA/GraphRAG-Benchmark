# GraphRAG-Bench — Génération Qwen sur TPU Colab

**Branche :** `feature/graphrag-pilot-llm-tpu` (issue de `feature/graphrag-pilot-llm-generation`). **Main inchangée.**

L'environnement Colab TPU affichant 47 Go de RAM et 225 Go de disque dispose de ressources hôtes importantes, mais **pas de CUDA**. Ces quantités ne renseignent pas sur la capacité HBM du TPU. Le code TPU de cette branche est distinct de la voie GPU T4.

## Backend

Utiliser TorchTPU (PyTorch `torch_tpu`) et Transformers v5, avec `--device tpu` et le modèle `Qwen/Qwen2.5-3B-Instruct`. L'inférence utilise `device_map="tpu"`, `dtype=torch.bfloat16`, la tokenisation en CPU, un padding à longueur de prompt fixe (3 584 tokens par défaut), et `cache_implementation="static"`. Les tenseurs d'entrée restent sur CPU pour la génération TPU, conformément au guide officiel Hugging Face. En cas de non-prise en charge de ce modèle par le backend TPU réel, **ne pas revenir silencieusement sur CPU**, utiliser la voie T4 validée par l'ancien notebook.

Sources : https://huggingface.co/docs/transformers/google_tpu et https://huggingface.co/blog/hf-torchtpu .

## Notebook prêt à lancer

`notebooks/GraphRAG_LLM_Pilot_TPU_Colab.ipynb`

1. Vérifier l'import `torch_tpu` et la disponibilité du `device="tpu"`.
2. Choisir le stockage temporaire ou Drive.
3. Cloner le fork et installer `transformers>=5.20,<6` avec `accelerate` en conservant TorchTPU.
4. Importer `GraphRAG_Comparaison.zip` (les fichiers de récupération partagée TF-IDF et BGE-large).
5. Exécuter neuf tests, puis le `--dry-run`.
6. Produire les deux réponses de la première question, sauvegarder après chaque réponse.
7. Produire les vingt-deux réponses restantes avec la même configuration, reprise par signature incluant le backend.
8. Télécharger les sorties JSON et CSV.

Les questions et les cinq passages sont les seuls contenus utilisés dans les prompts ; les réponses de référence sont réservées au CSV d'évaluation, non au LLM.

Les résultats ne sont pas évalués automatiquement et les annotations de preuves de l'étape précédente restent provisoires.

## Limitations

- **L'exécution réelle sur le TPU précis de l'utilisateur n'est pas encore qualifiée**, bien que le code ait une couverture de tests hors accélérateur.
- Le délai de première compilation TPU peut être important ; éviter de lancer vingt-quatre générations avant le premier test réussi.
- La session TPU gratuite peut s'interrompre. Le stockage `/content` est éphémère ; activer explicitement Drive si une sauvegarde durable est nécessaire.
- Modèle Qwen2.5-3B-Instruct : licence `qwen-research` (consulter les conditions avant diffusion/utilisation commerciale).

## Correctif du conflit PJRT (TPU déjà utilisé)

Si Colab affiche `RuntimeError: ... The TPU is already in use by process with pid ...`, vérifier si le **noyau Colab** a déjà initialisé `torch_tpu` en créant un tenseur `device="tpu"`. Dans ce cas, **ne pas lancer la génération dans `subprocess.run(BASE)`** : le sous-processus ne peut pas prendre le TPU occupé par le noyau. La cellule de génération du notebook mise à jour importe `load_inference`, `generate_paired` et `export_review` et utilise ces fonctions **dans le processus du noyau Colab**. Le modèle est gardé en mémoire et réutilisé pour les 24 réponses. Il ne faut pas supprimer le processus noyau Colab. La disparition du conflit PJRT ne garantit pas la compatibilité complète de Qwen : qualifier le premier essai avant la génération totale.
