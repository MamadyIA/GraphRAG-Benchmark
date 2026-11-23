# Prototype 1 — récupération du pilote, avant génération

Ce prototype dépend de la branche d'audit `audit/graphrag-bench-data-quality` et de son fichier `pilot_12_questions.json` produit localement. Il ne modifie pas le dataset.

## Périmètre

- Charge les corpus officiels JSON en mémoire, en séparant **médical** et **chaque roman** par sa source.
- Charge exclusivement le manifeste de 12 questions auditées.
- Découpe chaque source uniquement à partir de son champ `context`. Ni `answer` ni `evidence` n'entre dans l'index.
- Produit pour chaque question un classement de passages (`context` et `retrieved_chunks`), mais **aucune réponse générée** (`generated_answer: null`).
- Les résultats ne doivent pas être transmis à `generation_eval.py` avant l'ajout d'une véritable étape de génération.

## Vérification hors ligne (PowerShell)

Depuis la racine du dépôt après récupération de la branche :

```powershell
py -3.12 -m unittest discover -s tests -p "test_pilot_retrieval.py" -v
py -3.12 experiments/pilot_retrieval.py --root . --pilot ../graphrag_audit_results/pilot_12_questions.json --output ../graphrag_audit_results/pilot_lexical_retrieval.json --mode lexical --top-k 5
```

Le mode `lexical` utilise TF-IDF cosinus avec la bibliothèque standard. **C'est un test fonctionnel de récupération, pas le RAG vectoriel de l'article**. Il découpe en groupes de 256 **mots** avec chevauchement 32 mots.

## Essai sémantique — étape ultérieure, avec accord sur les ressources

Lancer dans un environnement virtuel Python distinct où `sentence-transformers` est installé :

```powershell
python experiments/pilot_retrieval.py --root . --pilot ../graphrag_audit_results/pilot_12_questions.json --output ../graphrag_audit_results/pilot_bge_retrieval.json --mode semantic --model BAAI/bge-large-en-v1.5 --top-k 5
```

Ce mode charge le modèle d'embeddings et peut nécessiter un téléchargement conséquent. Il découpe selon le tokenizer du modèle (256 **tokens**, chevauchement 32 tokens). Les similarités utilisent des vecteurs normalisés et un produit scalaire.

Les résultats des deux modes ne sont **pas comparables directement** sans harmonisation des unités de segmentation. Le modèle BGE demande également une qualification de sa configuration, de ses coûts et de sa performance locale avant un benchmark complet.

## Limitations

- Aucun LLM, aucune génération de réponse, aucune métrique calculée.
- `--top-k` sélectionne les premiers passages, sans reranker.
- `context` est une liste de chaînes pour faciliter l'évaluation des passages, mais l'interface officielle sera à valider avant branchement automatique.
- Modèle sémantique non testé sur GPU/CPU réel à ce stade.
- Vérification humaine des réponses et preuves du pilote encore à faire.
- Les fichiers de sorties sont écrits en dehors du dépôt par défaut dans les commandes recommandées.
