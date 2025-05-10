
---

### Phase 10: CI/CD Automation (Conceptual - GitHub Actions)

**1. Goal of this Phase:**
To define and conceptually outline an automated Continuous Integration/Continuous Deployment (CI/CD) pipeline using GitHub Actions. This pipeline would automate the process of testing, building the Docker image for our HVS prediction API, pushing the image to a container registry, and deploying the updated application to a Kubernetes cluster whenever changes are made to the codebase.

**2. Rationale (Why this phase is critical):**
CI/CD is a cornerstone of modern software development and MLOps, enabling rapid, reliable, and repeatable delivery of software.
* **Automation:** Significantly reduces the manual effort involved in the release process, minimizing the chances of human error.
* **Consistency:** Ensures that every code change goes through the same standardized build, test, and deployment process, leading to more predictable outcomes.
* **Speed & Frequency:** Allows for faster and more frequent updates to the application (e.g., deploying new model versions, bug fixes, or feature enhancements to the API).
* **Early Bug Detection:** Integrating automated tests into the pipeline means issues are caught earlier in the development cycle, making them easier and cheaper to fix.
* **Improved Developer Productivity:** Developers can focus on writing code and features, knowing that the CI/CD pipeline will handle the repetitive and operational aspects of getting their changes deployed.
* **Reproducibility:** The entire build and deployment process is defined as code (in a workflow YAML file), making it versionable and reproducible.
This phase directly addresses your project goal: *"Automated the end-to-end ML lifecycle using GitHub Actions for CI/CD, including automated testing, Docker image building/pushing (Docker Hub), and Kubernetes deployments (`kubectl apply`) triggered by code changes."*

**3. Approach & Key Activities (Defining the CI/CD Workflow):**
The core activity is to define a GitHub Actions workflow in a YAML file, typically located at `.github/workflows/cicd.yaml` in the project repository. This workflow specifies the events that trigger it and the jobs and steps to execute.

* **Workflow Triggers (`on`):**
    * The pipeline would typically be triggered on:
        * `push` events to the `main` branch (for deploying to a staging/production-like environment).
        * `pull_request` events targeting the `main` branch (for running tests and linting on proposed changes before merging).
* **Jobs:** The workflow would consist of several jobs that can run sequentially (if dependent) or in parallel.
    * **Job 1: Lint & Test (`lint-test`):**
        1.  **Checkout Code:** Fetches the latest code from the repository onto the GitHub Actions runner (a virtual machine).
        2.  **Setup Python:** Configures the desired Python version.
        3.  **Install Dependencies:** Installs packages from `requirements.txt` and any testing/linting tools (e.g., `flake8`, `pytest`).
        4.  **Linting:** Runs code linters (e.g., `flake8 .`) to check for style issues and basic errors.
        5.  **Automated Tests:** Executes unit and integration tests (e.g., `pytest tests/unit/ tests/integration/`). *(Note: We haven't written these tests in detail during our walkthrough, but a production project would have a comprehensive test suite.)* This job would fail if linting or tests fail, preventing the pipeline from proceeding.
    * **Job 2: Build and Push Docker Image (`build-push-image`):**
        * **Dependency:** This job would `need` the `lint-test` job to complete successfully.
        * **Conditional Execution:** Typically runs only on pushes to the `main` branch (not on pull requests).
        1.  **Checkout Code.**
        2.  **Login to Container Registry:** Authenticates with a Docker image registry (e.g., Docker Hub, Google Container Registry, AWS ECR). This requires storing registry credentials as encrypted **secrets** in the GitHub repository settings.
        3.  **Set up Docker Buildx:** (Optional but recommended) A Docker CLI plugin for extended build capabilities.
        4.  **Build Docker Image:** Builds the image using the `Dockerfile` in the project root.
        5.  **Tag Docker Image:** Tags the image, typically with:
            * The Git commit SHA (e.g., `your-repo/hvs-prediction-api:abcdef123`) for precise versioning.
            * A static tag like `latest` or a version number (e.g., `your-repo/hvs-prediction-api:v1.1`).
        6.  **Push Docker Image:** Pushes the tagged image(s) to the container registry.
    * **Job 3: Deploy to Kubernetes (`deploy-to-kubernetes`):**
        * **Dependency:** This job would `need` the `build-push-image` job to complete successfully.
        * **Conditional Execution:** Typically runs only on pushes to the `main` branch.
        1.  **Checkout Code.**
        2.  **Configure `kubectl`:** Sets up `kubectl` on the runner to communicate with the target Kubernetes cluster. This is a critical step and requires secure authentication, typically using credentials (kubeconfig file content or service account tokens) stored as GitHub secrets. The exact method depends on the Kubernetes provider (GKE, EKS, AKS, self-hosted).
        3.  **Update Kubernetes Manifests (Dynamic Image Tag):**
            * The `k8s/deployment.yaml` file needs to be updated to use the newly built and pushed Docker image tag (e.g., the Git commit SHA). This can be done using tools like `sed`, `yq` (a YAML processor), or `kustomize` directly within the workflow to modify the `image:` field in the deployment manifest.
        4.  **Apply Kubernetes Manifests:** Uses `kubectl apply -f k8s/service.yaml` and `kubectl apply -f k8s/deployment.yaml` (with the updated image tag) to roll out the new version of the application to the Kubernetes cluster.
        5.  **Verify Rollout (Optional):** Can include a step like `kubectl rollout status deployment/hvs-api-deployment` to wait for the deployment to complete successfully.

**4. Technical Insights & Decisions:**
* **Choice of CI/CD Tool (GitHub Actions):** GitHub Actions is a natural choice for projects hosted on GitHub due to its tight integration, ease of setup for basic workflows, and generous free tier for public repositories. Other tools include GitLab CI/CD, Jenkins, CircleCI, etc.
* **Secrets Management:** Critical for security. Credentials for Docker Hub, Kubernetes cluster access, and any other services must be stored as encrypted secrets in the GitHub repository settings (`Settings > Secrets and variables > Actions`) and accessed in the workflow using `${{ secrets.YOUR_SECRET_NAME }}`.
* **Image Tagging Strategy:** Tagging Docker images with the Git commit SHA provides excellent traceability and allows for easy rollbacks to specific code versions. Also tagging with `latest` or a semantic version (e.g., `v1.0.0`) is common.
* **Dynamic Manifest Updates:** The method chosen to update the image tag in `deployment.yaml` during the CI/CD run (e.g., `sed`, `yq`, `kustomize`) needs to be robust. `kustomize` is often preferred for more complex Kubernetes configuration management.
* **Environment-Specific Deployments:** For real projects, you'd typically have different workflows or conditional steps for deploying to different environments (e.g., development, staging, production), possibly triggered by pushes to different branches or by using tags.
* **Testing is Key:** The "CI" part (Continuous Integration) heavily relies on having a good suite of automated tests. If tests are flaky or insufficient, the pipeline might deploy buggy code.
* **ML Model in CI/CD:** This conceptual pipeline focuses on deploying the API application. A more advanced MLOps CI/CD pipeline would also include steps for:
    * **Model Retraining:** Triggering the training pipeline (Phase 4/5 scripts) when new data is available or drift is detected.
    * **Model Evaluation & Validation:** Automatically evaluating the retrained model and comparing it against the currently deployed one.
    * **Model Registration:** Pushing the validated model to a model registry (like MLflow Model Registry).
    * **Updating the API to use the new model:** This might involve updating the `BEST_MLFLOW_RUN_ID` in the deployment configuration and re-deploying the API.

**5. Key Commands/Code Snippets (Conceptual from `.github/workflows/cicd.yaml`):**
*(Refer to the previous detailed Phase 10 explanation for the full conceptual YAML - immersive ID `cicd_workflow_v1`)*

* **Trigger definition:**
  ```yaml
  on:
    push:
      branches: [ main ]
  ```
* **Checking out code:**
  ```yaml
  - name: Checkout code
    uses: actions/checkout@v4
  ```
* **Building and pushing Docker image (using a pre-built action):**
  ```yaml
  - name: Build and push Docker image
    uses: docker/build-push-action@v5
    with:
      push: true
      tags: yourusername/hvs-prediction-api:${{ github.sha }}
  ```
* **Applying Kubernetes manifest:**
  ```yaml
  - name: Deploy to Kubernetes
    run: |
      # (after configuring kubectl and updating deployment.yaml)
      kubectl apply -f k8s/deployment.yaml
      kubectl rollout status deployment/hvs-api-deployment
  ```

**6. Outcome of Phase 10 (Conceptual):**
* A defined CI/CD workflow (e.g., in `.github/workflows/cicd.yaml`) that conceptually automates the linting, testing, Docker image building/pushing, and Kubernetes deployment processes.
* An understanding of how to use GitHub Actions (or a similar CI/CD tool) to streamline the deployment lifecycle.
* Identification of the need for secure secret management for registry and cluster credentials.
* A blueprint for ensuring consistent and repeatable deployments triggered by code changes.

This phase conceptually ties together many of the previous development efforts into an automated release process, which is fundamental for efficient MLOps. While we haven't practically implemented all the secrets and cluster connections here, defining the workflow is the crucial design step.

---