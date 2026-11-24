# GraphRAG-Bench — qualification des preuves du pilote

## Objectif et statut

Calculer des métriques descriptives de couverture des preuves à partir de **passages préalablement annotés** pour les 12 questions pilotes. Les deux récupérateurs TF-IDF et BGE-large utilisent le **même découpage** de 256 tokens, recouvrement 32 tokens, Top-5.

**Les premières pré-annotations sont assistées par IA, non validées par un humain.** Aucun score n'est une métrique officielle de GraphRAG-Bench ; aucune conclusion statistique sur le corpus entier n'est permise.

Le jeu de données original n'est pas modifié et les sorties restent **hors du dépôt**.

## Livrables fournis séparément

L'archive `GraphRAG_Qualification_Preliminaire.zip` contient :
- `pilot_12_questions.json` : questions auditées, seed 42 ;
- `pilot_evidence_rubric.json` : facettes dérivées de `evidence`, avec périmètre `source`, `question_given` ou `interpretive` ;
- `pilot_evidence_ai_preannotations.csv` : 120 passages (12 × 5 × 2) avec une première annotation par facette, justification et statut `ai_prelabel` ;
- `pilot_evidence_provisional_metrics.json` : premières mesures ;
- `rapport_qualification_preuves.md` : contexte, résultats et limites.

**Ne pas ajouter les textes de passages complets dans le dépôt GitHub sans examen des droits et de la politique de diffusion des sources.**

## Évaluation dans Colab ou Python local

Dans le répertoire contenant les fichiers de l'archive :

```bash
python -m unittest -v test_evidence_eval.py
python evidence_eval.py \
  --pilot pilot_12_questions.json \
  --rubric pilot_evidence_rubric.json \
  --annotations pilot_evidence_ai_preannotations.csv \
  --output metrics_recomputed.json
```

Depuis la branche GitHub, utiliser plutôt :

```bash
python -m unittest discover -s tests -p "test_evidence_eval.py" -v
python experiments/evidence_eval.py \
  --pilot /chemin/pilot_12_questions.json \
  --rubric /chemin/pilot_evidence_rubric.json \
  --annotations /chemin/pilot_evidence_ai_preannotations.csv \
  --output /chemin/metrics_recomputed.json
```

## Définition des étiquettes

Chaque ligne contient `facet_ids_yes`, `facet_ids_partial`, `candidate_label`, `candidate_reason_fr`, `review_status`.
Les index de facettes commencent à 1 dans le fichier rubric. Seules les facettes `source` sont mesurées.

- `yes` : le passage contient au moins une preuve explicitement soutenue (chaque index mentionné dans `facet_ids_yes`).
- `partial` : preuve contextuelle ou incomplète, non comptée comme preuve stricte (`facet_ids_partial`).
- `no` : aucun support explicite n'a été identifié lors de la prélecture.
- `ai_prelabel` : jugement provisoire du modèle, **non validé**.
- `human_validated` : ne renseigner que lorsque le passage **a réellement été relu et arbitré** par un relecteur qualifié.

### Conditions des résultats finalisés

Après vérification humaine des 120 lignes, remplacer `review_status` par `human_validated` uniquement sur les lignes effectivement contrôlées. Utiliser alors :

```bash
python experiments/evidence_eval.py \
  --pilot /chemin/pilot_12_questions.json \
  --rubric /chemin/pilot_evidence_rubric.json \
  --annotations /chemin/annotations_validees.csv \
  --output /chemin/metrics_validees.json \
  --require-human-validated
```

Le script refuse toute annotation marquée `ai_prelabel` en mode final.

## Métriques descriptives

1. **Any-evidence Hit@5** : fraction de questions avec au moins un passage qui soutient explicitement une facette `source`.
2. **Full-facet-coverage@5** : fraction de questions dont toutes les facettes `source` sont couvertes par l'union des cinq passages.
3. **Strict facet recall@5** : nombre total de facettes `source` distinctes couvertes / nombre total de facettes `source`.
4. **Strict relevant-chunk precision@5** : passages au Top-5 soutenant au moins une facette `source` / total des passages.
5. **MRR@5** : rang réciproque moyen du premier passage soutenant au moins une facette `source`.

Les passages partiels ne comptent **pas** comme succès strict. Le rappel ne comptabilise une même facette qu'une seule fois même si plusieurs passages la répètent.

## Risques méthodologiques à arbitrer

- `Novel-506824e8` : la question parle de conflits avec les Sioux, alors que la réponse de référence cite la bataille de Mill Springs ; cohérence à contrôler.
- `Novel-980656b1` : réponse créative, plusieurs éléments inférés, source historique marquée par des préjugés coloniaux.
- `Medical-f847846c` : âge, symptômes et autres éléments sont déjà **dans la question** ; les exiger à nouveau dans un passage serait incorrect.
- `Novel-9fdb8f3a` : la troisième facette (« common ground for character interactions ») est interprétative et demande une règle d'arbitrage spécifique.
- L'ensemble du jeu pilote est limité à 12 questions équilibrées : éviter toute déclaration de significativité.
- Les quatre exclusions relevées par l'audit structurel ne garantissent pas l'exactitude des autres références.

## Décision

**GO** pour poursuivre la recherche technique et la préparation d'un générateur commun. **NO-GO** pour annoncer une supériorité scientifique de BGE-large sans revue humaine des preuves et sans dispositif expérimental plus étendu.
