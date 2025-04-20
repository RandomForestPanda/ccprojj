import time
import os
import shutil
import subprocess
import docker
import signal
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict
from db import Base, engine, SessionLocal, Function
from fastapi.middleware.cors import CORSMiddleware
from docker.errors import DockerException
from collections import defaultdict
from threading import Lock
import tarfile
import io, threading
from fastapi.requests import Request
import resource
from db import FunctionMetrics

from datetime import datetime, timedelta



global_container_pool = []
POOL_SIZE = 5
POOL_LOCK = Lock()

# In-memory container pool: maps image_tag → list of WarmContainer objects
warm_container_pool = defaultdict(list)

# How long to keep containers warm (in seconds)
WARM_EXPIRY_SECONDS = 300  # 5 minutes

class WarmContainer:
    def __init__(self, container, timestamp):
        self.container = container
        self.timestamp = timestamp

# Create the database tables
Base.metadata.create_all(bind=engine)
app = FastAPI()
# Add this middleware after creating the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Create code directory if it doesn't exist
CODE_DIR = "code"
os.makedirs(CODE_DIR, exist_ok=True)

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Schemas
class FunctionBase(BaseModel):
    name: str
    route: str
    language: str
    timeout: float
    code_path: str

class FunctionCreate(FunctionBase):
    code: str  # Actual code content

class FunctionUpdate(BaseModel):
    route: Optional[str] = None
    language: Optional[str] = None
    timeout: Optional[float] = None
    code: Optional[str] = None  # Updated code content

class FunctionOut(FunctionBase):
    id: int
    
    class Config:
        from_attributes = True

class ExecuteMultipleFunctionsRequest(BaseModel):
    function_ids: List[int]

def get_file_extension(language):
    """Get file extension based on language"""
    ext_map = {
        "python": ".py",
        "javascript": ".js"
    }
    return ext_map.get(language.lower(), ".txt")

def save_code_to_file(func_id, func_name, code_content, language):
    func_dir = os.path.join(CODE_DIR, str(func_id))
    os.makedirs(func_dir, exist_ok=True)
    ext = get_file_extension(language)
    # Change filename to always "main{ext}"
    filename = f"main{ext}"
    file_path = os.path.join(func_dir, filename)
    with open(file_path, "w") as f:
        f.write(code_content)
    return file_path

# Docker client
def initialize_docker_client():
    """Initialize the Docker client and handle errors gracefully."""
    try:
        client = docker.from_env()
        # Test Docker connection
        client.ping()
        return client
    except (DockerException, FileNotFoundError) as e:
        print("Warning: Docker is not available. Ensure Docker is running and accessible.")
        print(f"Error: {e}")
        return None

docker_client = initialize_docker_client()

def build_docker_image(func_id, func_name, language):
    """Build a Docker image for the function."""
    if not docker_client:
        raise HTTPException(status_code=500, detail="Docker is not available. Please start Docker.")
    
    func_dir = os.path.join(CODE_DIR, str(func_id))  # Directory for the function
    os.makedirs(func_dir, exist_ok=True)  # Ensure the directory exists
    
    dockerfile_template = f"Dockerfile.{language.lower()}"  # Template based on language
    dockerfile_path = os.path.join(func_dir, "Dockerfile")  # Destination Dockerfile path
    
    # Ensure the Dockerfile template exists
    if not os.path.exists(dockerfile_template):
        raise HTTPException(status_code=500, detail=f"Dockerfile template for {language} not found.")
    
    # Copy the Dockerfile template to the function's directory
    shutil.copy(dockerfile_template, dockerfile_path)
    
    # Removed renaming code as the function code is now stored as main{ext}
    image_tag = f"{func_name.lower()}_{func_id}"
    
    try:
        docker_client.images.build(path=func_dir, dockerfile="Dockerfile", tag=image_tag)
        return image_tag
    except docker.errors.BuildError as e:
        raise HTTPException(status_code=500, detail=f"Docker build failed: {str(e)}")

'''def run_function_in_docker(image_tag, timeout):
    """Run the function inside a Docker container with timeout enforcement."""
    if not docker_client:
        raise HTTPException(status_code=500, detail="Docker is not available. Please start Docker.")
    try:
        print(f"Running container for image: {image_tag} with timeout: {timeout}")
        container = docker_client.containers.run(image_tag, detach=True)
        print(f"Container {container.id} started.")
        
        result = container.wait(timeout=timeout)
        logs = container.logs().decode("utf-8")
        print(f"Container logs: {logs}")
        
        container.remove()
        print(f"Container {container.id} removed.")
        
        # Check for execution errors
        if result["StatusCode"] != 0:
            raise HTTPException(status_code=500, detail=f"Function execution failed with status code {result['StatusCode']}: {logs}")
        
        return logs
    except docker.errors.ContainerError as e:
        print(f"ContainerError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Function execution failed: {str(e)}")
    except docker.errors.APIError as e:
        print(f"APIError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
    except subprocess.TimeoutExpired:
        print("TimeoutExpired: Function execution timed out")
        raise HTTPException(status_code=408, detail="Function execution timed out")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")'''


def run_with_timeout(container, cmd, timeout):
    result = {
        'output': '',
        'exit_code': 1,
        'cpu_time': 0,
        'memory_used': 0,
        'execution_time': 0
    }

    def target():
        try:
            start_time = time.time()

            # Get resource usage before execution
            usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)

            # Execute the command
            exec_result = container.exec_run(cmd)

            # Get resource usage after execution
            usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
            end_time = time.time()

            # Calculate metrics
            cpu_time = usage_end.ru_utime - usage_start.ru_utime
            memory_used_kb = usage_end.ru_maxrss  # in kilobytes
            execution_time = end_time - start_time

            # Store results
            result.update({
                'output': exec_result.output.decode("utf-8") if exec_result.output else "",
                'exit_code': exec_result.exit_code,
                'cpu_time': cpu_time,
                'memory_used': memory_used_kb,
                'execution_time': execution_time
            })

        except Exception as e:
            result['error'] = str(e)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        container.kill()
        raise HTTPException(status_code=408, detail="Function execution timed out")

    if 'error' in result:
        raise HTTPException(status_code=500, detail=f"Execution error: {result['error']}")

    return result

      
def make_tar_archive(src_path, dest_filename):
    """Create a tar archive containing the function file (for Docker copy)."""
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        tar.add(src_path, arcname=dest_filename)
    data.seek(0)
    return data

def run_function_in_docker(code_path, language, timeout, runtime):
    """Run a function using a pre-warmed shared Docker container based on runtime and language."""
    if not docker_client:
        raise HTTPException(status_code=500, detail="Docker is not available.")

    with POOL_LOCK:
        # Ensure pool exists for the given language and runtime
        if language not in WARM_CONTAINER_POOLS or runtime not in WARM_CONTAINER_POOLS[language]:
            raise HTTPException(status_code=400, detail=f"Unsupported language or runtime: {language}, {runtime}")

        # Get container pool
        if not WARM_CONTAINER_POOLS[language][runtime]:
            raise HTTPException(status_code=503, detail=f"No available {language} containers for runtime {runtime}.")

        warm = WARM_CONTAINER_POOLS[language][runtime].pop(0)
        print(f"🚀 Assigned {language} container [{runtime}]: {warm.container.id[:12]} to run the function.")

    try:
        # Determine destination filename
        ext = ".py" if language == "python" else ".js"
        dest_filename = f"main{ext}"

        # Copy code into container
        archive = make_tar_archive(code_path, dest_filename)
        warm.container.put_archive("/mnt", archive)

        # Build command
        exec_cmd = f"python3 /mnt/{dest_filename}" if language == "python" else f"node /mnt/{dest_filename}"

        # Execute with timeout
        result = run_with_timeout(warm.container, exec_cmd, timeout)
        output = result['output']
        exit_code = result['exit_code']

        # Re-add container to pool
        with POOL_LOCK:
            WARM_CONTAINER_POOLS[language][runtime].append(WarmContainer(warm.container, time.time()))

        if exit_code != 0:
            raise HTTPException(status_code=500, detail=f"Function exited with code {exit_code}: {output}")

        return output

    except Exception as e:
        warm.container.remove(force=True)
        return f"Error running code in warm container: {str(e)}"


def store_metrics(db: Session, function_id: int, metrics: dict, runtime: str):
    """Store execution metrics in the database."""
    db_metrics = FunctionMetrics(
        function_id=function_id,
        execution_time=metrics['execution_time'],
        cpu_time=metrics['cpu_time'],
        memory_used=metrics['memory_used'],
        exit_code=metrics['exit_code'],
        runtime=runtime,
        success=metrics['exit_code'] == 0
    )
    db.add(db_metrics)
    db.commit()
    return db_metrics


def get_aggregated_metrics(db: Session, function_id: int, time_window: str = "24h"):
    """Get aggregated metrics for a function."""
    now = datetime.utcnow()

    if time_window == "1h":
        delta = timedelta(hours=1)
    elif time_window == "24h":
        delta = timedelta(days=1)
    elif time_window == "7d":
        delta = timedelta(days=7)
    else:
        delta = timedelta(days=1)  # default

    # Query metrics for the time window
    metrics = db.query(FunctionMetrics).filter(
        FunctionMetrics.function_id == function_id,
        FunctionMetrics.timestamp >= now - delta
    ).all()

    if not metrics:
        return None

    # Calculate aggregates
    execution_times = [m.execution_time for m in metrics]
    cpu_times = [m.cpu_time for m in metrics]
    memory_used = [m.memory_used for m in metrics]
    successes = sum(1 for m in metrics if m.success)
    failures = len(metrics) - successes

    return {
        "function_id": function_id,
        "time_window": time_window,
        "count": len(metrics),
        "success_rate": successes / len(metrics) if metrics else 0,
        "avg_execution_time": sum(execution_times) / len(execution_times),
        "max_execution_time": max(execution_times),
        "min_execution_time": min(execution_times),
        "avg_cpu_time": sum(cpu_times) / len(cpu_times),
        "avg_memory_used": sum(memory_used) / len(memory_used),
        "max_memory_used": max(memory_used),
        "successes": successes,
        "failures": failures
    }


@app.on_event("startup")
def initialize_global_container_pool():
    base_images = {
        "python": "python:3.10-slim",
        "javascript": "node:18-slim"
    }

    runtimes = ["runc", "runsc"]
    containers_per_combination = 2  # 2 per language per runtime
    now = time.time()

    global WARM_CONTAINER_POOLS
    WARM_CONTAINER_POOLS = {
        lang: {runtime: [] for runtime in runtimes}
        for lang in base_images
    }

    for lang, image in base_images.items():
        for runtime in runtimes:
            for i in range(containers_per_combination):
                try:
                    container = docker_client.containers.run(
                        image,
                        detach=True,
                        tty=True,
                        command="tail -f /dev/null",
                        runtime=runtime
                    )
                    WARM_CONTAINER_POOLS[lang][runtime].append(WarmContainer(container, now))
                    print(f"✅ {lang.capitalize()} container [{runtime}] {i+1} started: {container.id[:12]}")
                except Exception as e:
                    print(f"❌ Failed to start {lang} container [{runtime}] {i+1}: {e}")



# Create a new function
@app.post("/functions/", response_model=FunctionOut)
def create_function(function: FunctionCreate, db: Session = Depends(get_db)):
    existing = db.query(Function).filter(Function.name == function.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Function with this name already exists")
    
    # Create function in database first to get ID
    db_function = Function(
        name=function.name,
        route=function.route,
        language=function.language,
        timeout=function.timeout,
        code_path=""  # Temporary, will update after saving file
    )
    
    db.add(db_function)
    db.commit()
    db.refresh(db_function)
    
    # Now save code to file using the function ID and name
    code_path = save_code_to_file(db_function.id, db_function.name, function.code, function.language)
    
    # Update the code_path in database
    db_function.code_path = code_path
    db.commit()
    db.refresh(db_function)
    
    return db_function

# Read all functions
@app.get("/functions/", response_model=List[FunctionOut])
def list_functions(db: Session = Depends(get_db)):
    return db.query(Function).all()

# Read a single function by ID
@app.get("/functions/{function_id}", response_model=FunctionOut)
def get_function(function_id: int, db: Session = Depends(get_db)):
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    return db_function

# Get function code
@app.get("/functions/{function_id}/code")
def get_function_code(function_id: int, db: Session = Depends(get_db)):
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    
    try:
        # Resolve absolute path from project directory based on stored path
        abs_path = db_function.code_path if os.path.isabs(db_function.code_path) else os.path.join(os.getcwd(), db_function.code_path)
        print("DEBUG: Resolved code file path:", abs_path)  # Debug logging
        
        # Fallback: if the resolved path does not exist, try using the standard naming (main{ext})
        if not os.path.exists(abs_path):
            ext = get_file_extension(db_function.language)
            fallback_path = os.path.join(os.getcwd(), "code", str(function_id), f"main{ext}")
            print("DEBUG: Fallback code file path:", fallback_path)
            abs_path = fallback_path
            if not os.path.exists(abs_path):
                raise HTTPException(status_code=404, detail=f"Code file not found at {abs_path}")
                
        with open(abs_path, "r", encoding="utf-8") as f:
            code_content = f.read()
        return {"name": db_function.name, "language": db_function.language, "code": code_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error accessing code file: {str(e)}")

# Update a function by ID
@app.put("/functions/{function_id}", response_model=FunctionOut)
def update_function(function_id: int, function_update: FunctionUpdate, db: Session = Depends(get_db)):
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    
    # Update database fields first
    update_data = function_update.dict(exclude_unset=True, exclude={"code"})
    for key, value in update_data.items():
        setattr(db_function, key, value)
    
    # Update code file if provided
    if function_update.code is not None:
        # If language was updated, we need the new language, otherwise use existing
        language = function_update.language if function_update.language else db_function.language
        
        # Save updated code to file
        code_path = save_code_to_file(function_id, db_function.name, function_update.code, language)
        db_function.code_path = code_path
    
    db.commit()
    db.refresh(db_function)
    return db_function

# Delete a function by ID
@app.delete("/functions/{function_id}")
def delete_function(function_id: int, db: Session = Depends(get_db)):
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")
    
    # Delete code directory
    func_dir = os.path.join(CODE_DIR, str(function_id))
    if os.path.exists(func_dir):
        shutil.rmtree(func_dir)
    
    # Delete function from database
    db.delete(db_function)
    db.commit()
    return {"detail": f"Function '{db_function.name}' deleted successfully"}

# Endpoint to execute a function
@app.post("/functions/{function_id}/execute_docker")
async def execute_function(request: Request, function_id: Optional[int] = None, db: Session = Depends(get_db)):
    if function_id is None:
        raise HTTPException(status_code=400, detail="Function ID must be provided")

    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")

    try:
        result = run_function_in_docker(db_function.code_path, db_function.language, db_function.timeout, 'runc')

        # Store metrics
        store_metrics(db, function_id, {
            'execution_time': result['execution_time'],
            'cpu_time': result['cpu_time'],
            'memory_used': result['memory_used'],
            'exit_code': result['exit_code']
        }, 'runc')

        return {"logs": result['output']}
    except HTTPException as e:
        # Store failure metrics
        store_metrics(db, function_id, {
            'execution_time': 0,
            'cpu_time': 0,
            'memory_used': 0,
            'exit_code': 1
        }, 'runc')
        raise e


@app.post("/functions/{function_id}/execute_gvisor")
async def execute_function(request: Request, function_id: Optional[int] = None, db: Session = Depends(get_db)):
    if function_id is None:
        raise HTTPException(status_code=400, detail="Function ID must be provided")

    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")

    try:
        result = run_function_in_docker(db_function.code_path, db_function.language, db_function.timeout, 'runsc')

        # Store metrics
        store_metrics(db, function_id, {
            'execution_time': result['execution_time'],
            'cpu_time': result['cpu_time'],
            'memory_used': result['memory_used'],
            'exit_code': result['exit_code']
        }, 'runsc')

        return {"logs": result['output']}
    except HTTPException as e:
        # Store failure metrics
        store_metrics(db, function_id, {
            'execution_time': 0,
            'cpu_time': 0,
            'memory_used': 0,
            'exit_code': 1
        }, 'runsc')
        raise e


@app.get("/functions/{function_id}/metrics")
def get_function_metrics(
        function_id: int,
        time_window: str = "24h",
        db: Session = Depends(get_db)
):
    """Get aggregated metrics for a function."""
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")

    metrics = get_aggregated_metrics(db, function_id, time_window)
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics found for this function")

    return metrics


@app.get("/functions/{function_id}/metrics/raw")
def get_raw_metrics(
        function_id: int,
        limit: int = 100,
        db: Session = Depends(get_db)
):
    """Get raw metrics data for a function."""
    db_function = db.query(Function).filter(Function.id == function_id).first()
    if db_function is None:
        raise HTTPException(status_code=404, detail="Function not found")

    metrics = db.query(FunctionMetrics).filter(
        FunctionMetrics.function_id == function_id
    ).order_by(
        FunctionMetrics.timestamp.desc()
    ).limit(limit).all()

    return metrics

@app.post("/functions/execute")
def execute_multiple_functions(request: ExecuteMultipleFunctionsRequest, db: Session = Depends(get_db)):
    results = []
    for function_id in request.function_ids:
        db_function = db.query(Function).filter(Function.id == function_id).first()
        if db_function is None:
            results.append({"id": function_id, "error": "Function not found"})
            continue
        
        try:
            # Build Docker image
            image_tag = build_docker_image(function_id, db_function.name, db_function.language)
            
            # Run function in Docker
            #logs = run_function_in_docker(image_tag, db_function.timeout)
            logs = run_function_in_docker(db_function.code_path, db_function.language, db_function.timeout)

            results.append({"id": function_id, "logs": logs})
        except HTTPException as e:
            results.append({"id": function_id, "error": e.detail})
    
    return {"results": results}
