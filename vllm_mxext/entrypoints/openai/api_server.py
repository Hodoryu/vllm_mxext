# Add imports at the top of the file
from vllm_mxext.monitoring.dashboard import DashboardManager
from fastapi.staticfiles import StaticFiles
import os

# Add LoRA management imports
from vllm_mxext.models import LoadLoRAAdapterRequest, UnloadLoRAAdapterRequest

# Add LoRA management endpoints

@router.post("/v1/load_lora_adapter")
async def load_lora_adapter(request: LoadLoRAAdapterRequest, raw_request: Request):
    """Load a LoRA adapter dynamically."""
    try:
        handler = models(raw_request)
        response = await handler.load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(), status_code=response.code)
        return JSONResponse(content={"status": "success", "message": f"LoRA adapter {request.lora_name} loaded successfully"})
    except Exception as e:
        logger.error(f"Error loading LoRA adapter: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )

@router.post("/v1/unload_lora_adapter")
async def unload_lora_adapter(request: UnloadLoRAAdapterRequest, raw_request: Request):
    """Unload a LoRA adapter."""
    try:
        handler = models(raw_request)
        response = await handler.unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(), status_code=response.code)
        return JSONResponse(content={"status": "success", "message": f"LoRA adapter {request.lora_name} unloaded successfully"})
    except Exception as e:
        logger.error(f"Error unloading LoRA adapter: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )

@router.get("/v1/lora_adapters")
async def list_lora_adapters(raw_request: Request):
    """List currently loaded LoRA adapters."""
    try:
        handler = models(raw_request)
        adapters = await handler.list_lora_adapters()
        return JSONResponse(content={"adapters": adapters})
    except Exception as e:
        logger.error(f"Error listing LoRA adapters: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )

# Modify the build_app function to integrate dashboard
def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(openapi_url=None,
                      docs_url=None,
                      redoc_url=None,
                      lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)
    
    # Add dashboard integration
    dashboard_manager = DashboardManager()
    dashboard_manager.setup_routes(app)
    
    # Mount static files for dashboard
    static_path = os.path.join(os.path.dirname(__file__), "..", "..", "monitoring", "static")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    # Store dashboard manager in app state for access by other components
    app.state.dashboard_manager = dashboard_manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = ErrorResponse(message=str(exc),
                            type="BadRequestError",
                            code=HTTPStatus.BAD_REQUEST)
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    return app
