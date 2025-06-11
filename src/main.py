import supervisely as sly
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import src.globals as g
from src.functions import (
    auto_update_all_embeddings,
    check_in_progress_projects,
    safe_check_autorestart,
)
from src.utils import run_safe

app = sly.Application()
server = app.get_server()

if sly.is_development():
    # This will enable Advanced Debugging mode only in development mode.
    # Do not need to remove it in production.
    sly.app.development.enable_advanced_debug()


try:
    scheduler = AsyncIOScheduler(
        job_defaults={
            "misfire_grace_time": 80,  # Allow jobs to be 2 minutes late
            "coalesce": True,  # Combine missed runs into a single run
        }
    )

    scheduler.add_job(
        run_safe,
        args=[safe_check_autorestart],
        max_instances=1,
    )

    scheduler.add_job(
        run_safe,
        args=[auto_update_all_embeddings],
        trigger="interval",
        minutes=g.UPDATE_EMBEDDINGS_INTERVAL,
        max_instances=1,  # Prevent overlapping job instances
    )

    scheduler.add_job(
        run_safe,
        args=[check_in_progress_projects],
        trigger="interval",
        hours=g.CHECK_INPROGRESS_INTERVAL,  # Check every 4 hours by default
        max_instances=1,  # Prevent overlapping job instances
    )

    @server.on_event("startup")
    def on_startup():
        sly.logger.info("Starting scheduler...")
        scheduler.start()
        sly.logger.info("Scheduler started successfully")

    app.call_before_shutdown(scheduler.shutdown)

except Exception as e:
    sly.logger.error(f"Error during initialization: {e}")
    raise


@server.get("/health")
async def health_check():
    return {"status": "healthy", "scheduler_running": scheduler.running}
