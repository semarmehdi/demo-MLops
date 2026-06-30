import logging
import os
import signal
import sys
import time

from dotenv import load_dotenv

# Grâce au __init__.py, l'importation est centralisée et ultra propre
from utils import extract_employees, load_employees, transform_employees

# Configuration globale des logs pour voir défiler les étapes
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Flag global basculé par les signaux d'arrêt (SIGTERM via `docker stop`, SIGINT via Ctrl+C)
_shutdown = False


def _handle_signal(signum, _frame):
    """Permet un arrêt propre entre deux cycles plutôt qu'une coupure brutale."""
    global _shutdown
    logging.warning("Signal %s reçu — arrêt programmé après le cycle courant.", signum)
    _shutdown = True


### un cycle complet du pipeline ###
def run_pipeline():
    logging.info("==================================================")
    logging.info("🚀 DÉBUT DU PIPELINE ETL (SANS AIRFLOW)")
    logging.info("==================================================")

    # 1. EXTRACT
    logging.info("--- Étape 1 : Extraction des données API & Backup S3 ---")
    raw_data = extract_employees()

    # 2. TRANSFORM
    logging.info("--- Étape 2 : Transformation & Prédiction du modèle ---")
    processed_df = transform_employees(raw_data)

    # 3. LOAD
    logging.info("--- Étape 3 : Chargement final (S3 Clean & Postgres) ---")
    load_summary = load_employees(processed_df)

    logging.info("✅ Cycle terminé. Résumé du chargement : %s", load_summary)
    return load_summary


### ordonnanceur "maison" qui simule un job de prod récurrent ###
def run_scheduler():
    """
    Variables d'environnement de contrôle :
      ETL_LOOP_ITERATIONS        Nombre de cycles à exécuter.
                                 <= 0  => boucle infinie (comportement prod).
                                 Défaut 1 (exécution unique = comportement d'origine).
      ETL_LOOP_INTERVAL_SECONDS  Pause entre deux cycles, en secondes. Défaut 30.
      ETL_FAIL_FAST              'true'  => arrêt au 1er échec (idéal pour la CI).
                                 'false' => on logue l'erreur et on continue
                                            (ordonnanceur résilient, comme en prod).
    """
    # Chargement des variables d'env une seule fois, au point d'entrée unique
    load_dotenv()
    logging.info("Variables d'environnement chargées avec succès.")

    iterations = int(os.getenv("ETL_LOOP_ITERATIONS", "1"))
    interval = float(os.getenv("ETL_LOOP_INTERVAL_SECONDS", "30"))
    fail_fast = os.getenv("ETL_FAIL_FAST", "true").lower() == "true"
    infinite = iterations <= 0

    logging.info(
        "🗓️  Ordonnanceur démarré | cycles=%s | intervalle=%ss | fail_fast=%s",
        "∞" if infinite else iterations,
        interval,
        fail_fast,
    )

    count = 0
    failures = 0
    while not _shutdown and (infinite or count < iterations):
        count += 1
        suffix = "" if infinite else f"/{iterations}"
        logging.info("######## CYCLE %s%s ########", count, suffix)

        try:
            run_pipeline()
        except Exception as e:
            failures += 1
            logging.error("❌ Cycle %s en échec : %s", count, e, exc_info=True)
            if fail_fast:
                logging.error("fail_fast=true => arrêt immédiat de l'ordonnanceur.")
                sys.exit(1)

        # On ne dort PAS après le dernier cycle (ni si un arrêt a été demandé)
        last_cycle = not infinite and count >= iterations
        if _shutdown or last_cycle:
            break

        logging.info("⏳ Attente de %ss avant le prochain cycle...", interval)
        # Sleep découpé par tranches d'1 s pour réagir vite à un signal d'arrêt
        slept = 0.0
        while slept < interval and not _shutdown:
            time.sleep(min(1.0, interval - slept))
            slept += 1.0

    logging.info("==================================================")
    logging.info(
        "🏁 Ordonnanceur terminé | cycles exécutés=%s | échecs=%s", count, failures
    )
    logging.info("==================================================")

    # En mode résilient (fail_fast=false) on signale quand même un code retour
    # non nul si au moins un cycle a échoué, sauf arrêt volontaire par signal.
    if failures and not _shutdown:
        sys.exit(1)


if __name__ == "__main__":
    # Arrêt propre sur `docker stop` (SIGTERM) ou Ctrl+C (SIGINT)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    run_scheduler()
