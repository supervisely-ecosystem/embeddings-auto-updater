import supervisely as sly
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import src.globals as g
from src.functions import auto_update_all_embeddings
from src.utils import run_safe

app = sly.Application()
server = app.get_server()

if sly.is_development():
    # This will enable Advanced Debugging mode only in development mode.
    # Do not need to remove it in production.
    sly.app.development.enable_advanced_debug()


scheduler = AsyncIOScheduler(
    job_defaults={
        "misfire_grace_time": 120,  # Allow jobs to be 2 minutes late
        "coalesce": True,  # Combine missed runs into a single run
    }
)
scheduler.add_job(
    run_safe,
    args=[auto_update_all_embeddings],
    trigger="interval",
    minutes=g.UPDATE_EMBEDDINGS_INTERVAL,
    max_instances=1,  # Prevent overlapping job instances
)


@server.on_event("startup")
def on_startup():
    scheduler.start()


app.call_before_shutdown(scheduler.shutdown)
