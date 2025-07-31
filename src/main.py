import datetime

import supervisely as sly
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi.responses import JSONResponse

import src.globals as g
from src.cas import is_flow_ready
from src.functions import (
    auto_update_all_embeddings,
    check_in_progress_projects,
    safe_check_autorestart,
    stop_embeddings_update,
)
from src.qdrant import client as qdrant_client
from src.utils import check_generator_is_ready, run_safe

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
        minutes=g.CHECK_INPROGRESS_INTERVAL,  # Check every 4 hours by default
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
    status = "healthy"
    checks = {}
    status_code = 200
    try:
        # Check Qdrant connection
        try:
            await qdrant_client.info()
            checks["qdrant"] = "healthy"
        except Exception as e:
            checks["qdrant"] = f"unhealthy: {str(e)}"
            status = "degraded"
            status_code = 503

        # Check CLIP service availability
        try:
            if await is_flow_ready():
                checks["clip"] = "healthy"
            else:
                checks["clip"] = "unhealthy: CLIP service is not ready"
                status = "degraded"
                status_code = 503
        except Exception as e:
            checks["clip"] = f"unhealthy: {str(e)}"
            status = "degraded"
            status_code = 503

        # Check Generator service availability
        try:

            if await check_generator_is_ready(endpoint=g.generator_host):
                checks["generator"] = "healthy"
            else:
                checks["generator"] = "unhealthy: Generator service is not ready"
                status = "degraded"
                status_code = 503
        except Exception as e:
            checks["generator"] = f"unhealthy: {str(e)}"
            status = "degraded"
            status_code = 503

        # Check if the scheduler is running
        try:
            if scheduler.running:
                checks["scheduler"] = "healthy"
            else:
                checks["scheduler"] = "unhealthy: Scheduler is not running"
                status = "degraded"
                status_code = 503
        except Exception as e:
            checks["scheduler"] = f"unhealthy: {str(e)}"
            status = "degraded"
            status_code = 503

    except Exception as e:
        status = "unhealthy"
        checks["general"] = f"error: {str(e)}"
        status_code = 500
    return JSONResponse(
        {
            "status": status,
            "checks": checks,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
        },
        status_code=status_code,
    )


@server.post("/stop-embeddings-update/{project_id}")
async def stop_update_project(project_id: int):
    """
    Stop auto_update_embeddings task for a specific project and reset all related states.

    :param project_id: Project ID to stop the embeddings update for
    :return: JSON response with operation status and details
    """
    try:
        result = await stop_embeddings_update(g.api, project_id)
        # Check if operation was successful or if there was nothing to stop
        operation_successful = result.get("success", False)
        if operation_successful:
            status_code = 200
        else:
            status_code = 400
        return JSONResponse(result, status_code=status_code)
    except Exception as e:
        sly.logger.error(f"Error in stop_embeddings_update_endpoint: {e}", exc_info=True)
        return JSONResponse(
            {
                "project_id": project_id,
                "success": False,
                "message": f"Internal server error: {str(e)}",
                "details": {},
            },
            status_code=500,
        )
