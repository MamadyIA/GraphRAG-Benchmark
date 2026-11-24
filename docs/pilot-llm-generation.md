# GraphRAG-Bench — pilote RAG génératif (Colab, Tesla T4)

## Objectif

Comparer **TF-IDF + Qwen2.5-3B-Instruct** et **BGE-large + Qwen2.5-3B-Instruct** sur les 12 questions pilotes, en utilisant les deux résultats de récupération **sur les mêmes segments de 256 tokens** (chevauchement 32, Top-5).

Le LLM et ses paramètres sont identiques pour les deux méthodes. La seule variable contrôlée est l'ensemble des passages classés.

## Accès

Ouvrir `notebooks/GraphRAG_LLM_Pilot_Colab.ipynb` depuis la branche `feature/graphrag-pilot-llm-generation` dans Google Colab et activer le GPU T4.

Le notebook utilise l'archive sauvegardée lors de l'expérience précédente (`GraphRAG_Comparaison.zip`) **si les deux JSON de récupération ne sont plus présents** dans la session Colab. Aucun texte de réponse de référence n'est chargé dans un prompt de génération.

## Entrées nécessaires

- `pilot_tfidf_shared_retrieval.json`
- `pilot_bge_shared_retrieval.json`

Ces deux fichiers doivent partager questions, sources et découpages. Le script refuse les divergences de références, de rangs, d'ID, de texte d'un même segment ou de sources. Les réponses de référence et preuves sont conservées uniquement comme métadonnées pour la revue hors prompt.

## LLM et paramètres

- Modèle : `Qwen/Qwen2.5-3B-Instruct` (téléchargement libre d'accès depuis Hugging Face ; **consulter sa licence qwen-research** avant réutilisation/distribution).
- Dispositif : GPU CUDA Colab, poids float16, Transformers + Accelerate.
- `do_sample=False` (génération gloutonne), `max_new_tokens=384`.
- Budget de prompt : `max_input_tokens=3584` ; le script échoue si le budget est dépassé au lieu de tronquer silencieusement les preuves.
- Langue : anglais, alignée avec les questions.
- Les passages sont délimités comme documents non fiables et cités `[P1]...[P5]`.
- Une graine 42 est fixée ; la reproductibilité bit à bit n'est pas garantie entre versions et architectures GPU.

## Vérification préalable (sans LLM)

```bash
python -m unittest discover -s tests -p "test_pilot_llm_generate.py" -v

python experiments/pilot_llm_generate.py \
  --retrieval-dir /content/graphrag_audit_results \
  --output /content/graphrag_llm_results/pilot_qwen25_3b_answers.json \
  --review /content/graphrag_llm_results/pilot_qwen25_3b_review.csv \
  --dry-run
```

Dix tests sont fournis et doivent réussir avant la qualification GPU, dont deux tests de compatibilité BatchEncoding/Tensor. Le dry run vérifie les fichiers sans télécharger de modèle.

## Essai pilote puis génération complète

Ajouter aux commandes ci-dessus (en retirant `--dry-run`) :

```bash
--model Qwen/Qwen2.5-3B-Instruct --device cuda --max-new-tokens 384 \
--max-input-tokens 3584 --limit-questions 1
```

Cette étape produit deux réponses à la première question : TF-IDF+Qwen et BGE+Qwen. Ensuite **relancer exactement la même commande sans** `--limit-questions 1` pour couvrir les douze questions.

Le programme écrit un checkpoint JSON après **chaque réponse** de façon atomique. En cas d'arrêt de la commande, relancer avec les mêmes entrées et paramètres pour reprendre les réponses manquantes. La reprise après réinitialisation du runtime nécessite d'avoir **sauvegardé le checkpoint dans Google Drive** (option proposée dans le notebook). Sans cela, le disque /content est éphémère.

## Sorties

- `pilot_qwen25_3b_answers.json` : état, empreintes de configuration/entrées, 24 réponses au maximum, traces de génération et passages utilisés (sous forme d'indices/hash).
- `pilot_qwen25_3b_review.csv` : 12 lignes avec références de vérité terrain, réponses des deux méthodes et champs d'arbitrage humain **vides**.
- `GraphRAG_LLM_Pilote.zip` : archive de sauvegarde préparée par le notebook.

**Aucune métrique de qualité n'est produite automatiquement par ce pipeline.** Qualité factuelle, fidélité aux cinq passages, pertinence des citations et complétude doivent être vérifiées par un évaluateur distinct.

## Limites et gouvernance

- **GO technique** pour l'essai de génération contrôlée, **pas un GO scientifique**.
- Les annotations des preuves de récupération (étape précédente) restent provisoires.
- Qwen2.5-3B-Instruct n'est pas nécessairement le modèle génératif utilisé dans l'article officiel : il s'agit d'une baseline expérimentale additionnelle, et non d'une réplication exacte de GraphRAG-Bench.
- Ne pas publier des réponses ou métriques sans vérifier les droits sur les corpus, les licences des modèles et les contraintes de licence du benchmark.
- Les scores ne doivent pas servir à inférer des conseils médicaux fiables ; les questions médicales sont traitées exclusivement comme benchmark.
- Aucune fusion de `main`, aucun déploiement et aucun paiement ne sont nécessaires.

## Correctif pour Transformers 5 : `BatchEncoding`

Sur des versions récentes de Transformers, `apply_chat_template(..., return_tensors="pt", return_dict=True)` renvoie un objet de type `BatchEncoding`, **pas un simple tenseur**. Le générateur utilise `input_ids` et le véritable `attention_mask` quand ils sont disponibles. Il préserve la limite de tokens et la génération déterministe. Il faut synchroniser le script et les tests depuis la branche GitHub ; remplacer uniquement `tokens.shape` par `tokens['input_ids'].shape` ne suffit pas car `torch.ones_like(tokens)` et `tokens.to(...)` doivent recevoir les bons tenseurs.

Lorsque l'ancienne version du script a été éditée manuellement dans Colab, la cellule 3 du notebook met les copies existantes de côté et récupère seulement les deux fichiers corrigés sans perdre les autres travaux. Le runtime gratuit T4 doit être réellement connecté et `torch.cuda.is_available()` doit renvoyer `True`.
