# LightRAG — qualification technique avant le pilote

**Statut : code préparé, exécution réelle GPU à qualifier.** Version retenue : lightrag-hku==1.4.11 (distincte de la version historique du benchmark). 

La première expérience est un smoke test sur un document fictif, sans réponses du benchmark ni API payante. Qwen2.5-3B-Instruct effectue l'extraction d'entités/relations et de mots-clés ; BGE-large calcule les embeddings. 12 appels LLM maximum, 1024 nouveaux tokens par appel d'indexation/requête. Le plafond de génération finale du pilote reste 384 tokens ; ces appels internes ne sont pas les réponses finales.

## Exécution Colab T4

```bash
pip install "lightrag-hku==1.4.11" sentence-transformers accelerate
python -m unittest discover -s tests -p "test_pilot_lightrag_smoke.py" -v
python -u experiments/pilot_lightrag_smoke.py --workdir /content/lightrag_smoke_new
```

Choisir un nouveau dossier à chaque tentative : le script préserve les anciennes sorties. La réussite exige un graphe GraphML non trivial et une récupération structurée contenant entités, relations et passages en mode hybrid. Les compteurs ne prouvent pas la justesse du graphe : relire les éléments extraits. Un échec ne doit pas être remplacé par une récupération vectorielle silencieuse.

## Avant la comparaison réelle

- Indexer chaque source complète et séparément (les cinq sources sollicitées par les 12 questions), sans consulter les réponses ou preuves de référence.
- Préparer un découpage partagé avec les baselines (256 tokens BGE, chevauchement 32), préserver les textes et identifiants des chunks ; le smoke utilise la tokenisation native de LightRAG, pas encore le découpage harmonisé.
- Vérifier la qualité et provenance des entités/relations, les coûts d'indexation et le budget de contexte. Le smoke utilise BGE sans préfixe de recherche ; aligner/documenter le traitement des embeddings avant la comparaison.
- Comparer la sélection de cinq passages LightRAG suivie du même prompt Qwen final que les baselines. Une variante native qui transmet également les entités et relations au générateur est une expérience distincte, à budget comparable.
- Réutiliser les 12 questions et le budget final initial de 384 tokens. Ne pas modifier les résultats existants ; signaler les trois réponses potentiellement tronquées. Prévoir une expérience de sensibilité distincte.
- Grille commune : exactitude, fidélité, complétude, citations étayées et qualité de l'abstention. L'annotation humaine et l'arbitrage de Novel-506824e8 restent nécessaires.

Le smoke test ne reproduit pas les métriques officielles et ne suffit pas pour annoncer une supériorité de GraphRAG.
