
---

### Phase 7: Containerization (Docker)

**1. Goal of this Phase:**
To package our Flask API application, along with all its Python dependencies, source code (`src/`), and necessary configurations, into a standardized, portable, and isolated unit called a **Docker container image**. This image can then be run consistently across different environments.

**2. Rationale (Why this phase is critical):**
Containerization with Docker is a foundational practice in modern software development and MLOps for several reasons:
* **Environment Consistency:** Docker ensures that the application runs in the exact same environment (OS, Python version, library versions) regardless of where the container is deployed – be it your local machine, a teammate's machine, a staging server, or a production Kubernetes cluster. This eliminates the common "it works on my machine" problem.
* **Dependency Management:** All dependencies are explicitly defined and bundled within the image via `requirements.txt`, making the application self-contained.
* **Isolation:** Containers run in isolated environments, preventing conflicts between dependencies of different applications running on the same host system.
* **Portability:** A Docker image can be easily stored in a container registry (like Docker Hub, Google Container Registry, AWS ECR) and pulled to run on any system that has Docker installed.
* **Scalability & Orchestration:** Containerized applications are designed to be managed by container orchestration platforms like Kubernetes (our Phase 8). Kubernetes excels at deploying, scaling, and managing containerized workloads.
* **Streamlined Development-to-Production:** Docker creates a consistent artifact (the image) that moves through different stages of the development lifecycle (dev, test, staging, prod).

**3. Approach & Key Activities:**
This phase involved creating two key configuration files (`Dockerfile` and `.dockerignore`) and using Docker commands to build the image and run it locally for testing.

* **`requirements.txt` Update:**
    * Ensured this file (in the project root) accurately listed all Python packages required by the API (Flask, Pydantic, MLflow, Scikit-learn, XGBoost, Pandas, NumPy, Joblib, `prometheus-flask-exporter`).
    * **Command (to generate/update):** `pip freeze > requirements.txt` (run in the activated `venv`, followed by manual review to remove dev-only packages).
* **Creating the `Dockerfile` (ID: `dockerfile_v1`):**
    * This text file, placed in the project root, contains step-by-step instructions for Docker to build the image.
    * **Key Instructions Used:**
        * `FROM python:3.11-slim`: Specified the base image. We chose a specific Python version (3.11) and a `-slim` variant to keep the image size relatively small.
        * `ENV PYTHONDONTWRITEBYTECODE 1`, `ENV PYTHONUNBUFFERED 1`: Common Python environment variables for containerized applications.
        * `WORKDIR /app`: Set the working directory inside the container to `/app`.
        * `COPY requirements.txt .`: Copied the requirements file into the image. This is done early to leverage Docker's layer caching – if `requirements.txt` doesn't change, the layer installing dependencies can be reused in subsequent builds, speeding them up.
        * `RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt`: Upgraded pip and installed all Python dependencies. `--no-cache-dir` helps reduce image size.
        * `COPY ./src ./src`: Copied our application's source code (the `src` directory containing `api/`, `services/`, etc.) into the `/app/src` directory within the image.
        * `ENV BEST_MLFLOW_RUN_ID="RUN_ID_NOT_SET"`: Defined a placeholder for the MLflow Run ID environment variable. The actual value is passed during `docker run` or via Kubernetes configuration.
        * `ENV MLFLOW_TRACKING_URI="file:///app/mlruns"`: **Crucial for MLflow in Docker.** This tells MLflow running *inside* the container where to find its local tracking data store root if we mount the `mlruns` directory.
        * `EXPOSE 5001`: Documented that the application inside the container will listen on port 5001.
        * `CMD ["python", "src/api/app.py"]`: Specified the default command to run when the container starts. This executes our Flask application's entry point. (Noted that for production, a WSGI server like Gunicorn would be preferred over the Flask development server).
* **Creating the `.dockerignore` file (ID: `dockerignore_v1`):**
    * This file, also in the project root, lists files and directories that should be *excluded* from the Docker build context (and thus not copied into the image).
    * **Key Exclusions:** `.git/`, `venv/`, `__pycache__/`, `notebooks/`, `data/` (except if small reference data was needed), `mlruns/` (as this is runtime data, mounted for local testing), `tests/` (unless needed for in-container tests), IDE-specific folders (`.vscode/`, `.idea/`).
    * **Rationale:** Keeps the image size smaller, improves build speed, and prevents sensitive or unnecessary files from being included in the image.
* **Building the Docker Image:**
    * **Command (from project root):**
        ```powershell
        docker build -t hvs-prediction-api:latest .
        ```
        * `-t hvs-prediction-api:latest`: Tags the image with the name `hvs-prediction-api` and the tag `latest`.
        * `.`: Specifies the current directory as the build context. Docker sends these files (respecting `.dockerignore`) to the Docker daemon to build the image.
* **Running the Docker Container Locally (for Testing):**
    * This step was iterative as we debugged MLflow model loading. The successful command involved:
        ```powershell
        docker run -p 5001:5001 `
                   -e BEST_MLFLOW_RUN_ID="YOUR_SELECTED_RUN_ID" `
                   -v ${PWD}/mlruns:/app/mlruns `
                   --name hvs-api `
                   hvs-prediction-api:latest
        ```
    * `-p 5001:5001`: Maps port 5001 on the host to port 5001 in the container.
    * `-e BEST_MLFLOW_RUN_ID="YOUR_SELECTED_RUN_ID"`: Sets the environment variable inside the container, which `prediction_service.py` uses to load the correct model.
    * `-v ${PWD}/mlruns:/app/mlruns`: **Crucial for local testing with local MLflow runs.** This mounts the `mlruns` directory from the host machine (your project root) into `/app/mlruns` inside the container. Combined with `ENV MLFLOW_TRACKING_URI="file:///app/mlruns"` in the Dockerfile, this allows MLflow inside the container to find and load the model artifacts. `${PWD}` in PowerShell refers to the current working directory.
    * `--name hvs-api`: Assigns a name to the running container for easier management.
* **Testing the Containerized API:**
    * Once the container was running, the `/predict` endpoint was tested using `Invoke-RestMethod` (or `curl`/Postman) against `http://localhost:5001/predict` with sample JSON payloads (`test_payload.json`, `test_payload_hazardous.json`).
    * The `/metrics` endpoint (`http://localhost:5001/metrics`) was also checked to ensure the Prometheus exporter was working.

**4. Technical Insights & Decisions:**
* **Base Image Choice (`python:3.11-slim`):** Using a specific version tag (`3.11`) ensures reproducibility. The `-slim` variant is smaller than the full Debian-based image, leading to a smaller final image size. Alpine-based Python images are even smaller but can sometimes lead to compatibility issues with libraries that have C dependencies if not handled carefully.
* **Leveraging Docker Cache:** Structuring the `Dockerfile` to copy `requirements.txt` and run `pip install` *before* copying the application source code (`COPY ./src ./src`) allows Docker to cache the dependency layer. If only the source code changes, Docker doesn't need to reinstall all dependencies, speeding up subsequent builds.
* **Environment Variables for Configuration:** Using environment variables (`BEST_MLFLOW_RUN_ID`, `MLFLOW_TRACKING_URI`) is a standard way to configure applications within containers. This makes the image more flexible, as the same image can be run with different configurations by just changing the environment variables at runtime.
* **Volume Mounting for Local MLflow Data:** The solution to mount the host's `mlruns` directory was a pragmatic approach for local development and testing when using MLflow's default file-based backend. For a production or shared environment, a centralized MLflow Tracking Server (with a database and artifact store like S3) would be used, and the API would be configured with its URI.
* **Debugging Path Issues:** The errors encountered (MLflow "Run not found" and "FileNotFoundError" for artifacts) highlighted the importance of understanding how paths are resolved across different environments (Windows host vs. Linux container) and how MLflow stores and retrieves artifact URIs. Setting `MLFLOW_TRACKING_URI` inside the container and using direct path construction in the service for loading (if `download_artifacts` proves tricky with local absolute paths) were key debugging steps.

**5. Key Commands:**
* **Update requirements:** `pip freeze > requirements.txt` (then manually review)
* **Build image:** `docker build -t hvs-prediction-api:latest .`
* **Run container (PowerShell example):**
  `docker run -p 5001:5001 -e BEST_MLFLOW_RUN_ID="your_run_id" -v ${PWD}/mlruns:/app/mlruns --name hvs-api hvs-prediction-api:latest`
* **Check container logs:** `docker logs hvs-api`
* **Stop container:** `docker stop hvs-api`
* **Remove container:** `docker rm hvs-api`
* **List running containers:** `docker ps`
* **List all images:** `docker images`

**6. Outcome of Phase 7:**
* A functional `Dockerfile` and `.dockerignore` file.
* A Docker image (`hvs-prediction-api:latest`) built, containing the Flask API and all its dependencies.
* Successful local execution and testing of the containerized API, including its ability to load the specified MLflow model/scaler (via the mounted `mlruns` volume) and serve predictions to `/predict`, as well as expose metrics on `/metrics`.
* The application is now packaged in a portable and consistent format, ready for deployment to a container orchestration platform like Kubernetes.

This phase successfully prepared our application for more scalable and robust deployments.

---