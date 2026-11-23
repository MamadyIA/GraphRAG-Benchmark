# GraphRAG-Bench: qualification des données

## Objectif

Auditer les JSON officiels **sans les modifier** et constituer un échantillon pilote reproductible de 12 questions.

Base de départ du fork : `fdbab5959b18c96532580877ffe27d112bccc0ec`.

Le rapport recherche : IDs dupliqués, champs obligatoires vides, catégories inconnues, sources non présentes dans les corpus et questions textuellement identiques ayant des réponses de référence différentes.

Le cas connu `Novel-55f0c0e2` apparaît deux fois (indices zéro-based 329 et 871), avec des sources et réponses différentes. **Aucune correction automatique de vérité terrain n'est effectuée.** Les enregistrements signalés sont exclus du pilote, en attente de validation de leurs preuves.

## Exécuter sous Windows PowerShell

Depuis la racine du dépôt, après avoir récupéré cette branche :

```powershell
py -3.12 -m unittest discover -s tests -p "test_benchmark_data_audit.py" -v
py -3.12 tools/benchmark_data_audit.py --root . --seed 42
```

Sans dépendance supplémentaire, sans réseau, sans API ni modèle. Le programme génère, **à côté du dépôt** (pas dedans) :

- `../graphrag_audit_results/data_quality_report.json` : diagnostics, index originaux, anomalies.
- `../graphrag_audit_results/pilot_12_questions.json` : 12 questions avec leur domaine, index original et champs officiels.

Sélection : six questions par domaine, soit par domaine 2 `Fact Retrieval`, 2 `Complex Reasoning`, 1 `Contextual Summarize`, 1 `Creative Generation`. La graine `42` fixe le tirage. Aucun fichier `Datasets/` n'est écrasé.

## Limites importantes

Ce contrôle est **structurel**, non une validation factuelle ou sémantique des questions, réponses et preuves. Une question admissible peut encore comporter une erreur. Le pilote est adapté à une vérification fonctionnelle, **pas** à des conclusions statistiques.

Les scripts officiels de modèles chargent leurs questions en Parquet. Le fichier pilote n'y est **pas encore intégré** ; aucune exécution de RAG, GraphRAG ou métrique n'a été réalisée.

Ne pas fusionner avec `main` sans tests et revue.