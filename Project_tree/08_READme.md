
---

### Phase 8: Local Kubernetes Deployment (Docker Desktop)

**1. Goal of this Phase:**
To deploy our containerized HVS prediction API onto a local Kubernetes cluster (specifically using the Kubernetes environment provided by Docker Desktop). This involves:
* Running multiple instances (replicas) of our API for simulated scalability and availability.
* Exposing the API as a network service within the Kubernetes cluster.
* Verifying that the deployed API is functional and accessible.

**2. Rationale (Why this phase is critical):**
While `docker run` is excellent for running single containers locally, Kubernetes is the industry standard for orchestrating containerized applications in more complex and production-like environments.
* **Scalability & High Availability:** Kubernetes allows us to easily run multiple identical instances (replicas) of our API. If one instance crashes, Kubernetes can restart it, and traffic can be directed to other healthy instances. This is crucial for building a resilient service, as mentioned in your initial project description (`replicas: 3`, high availability, scalability, fault tolerance).
* **Automated Management:** Kubernetes handles tasks like scheduling containers onto nodes (in Docker Desktop, there's one node), managing their lifecycle, and rolling out updates.
* **Service Discovery & Load Balancing:** Kubernetes provides a stable way (Services) to access our application pods, even if the pods' internal IP addresses change or they are rescheduled. It also distributes network traffic among the available replicas.
* **Foundation for Production:** Learning to deploy on a local Kubernetes cluster like Docker Desktop's is a valuable stepping stone towards deploying on cloud-managed Kubernetes services (GKE, EKS, AKS) or self-managed production clusters. It allows us to define our application's desired state using declarative manifest files.

**3. Approach & Key Activities:**
This phase involved enabling Kubernetes in Docker Desktop, ensuring our Docker image was accessible, updating our Kubernetes manifest files (`deployment.yaml` and `service.yaml`) with necessary configurations (especially for `mlruns` access), applying these manifests, and then verifying and testing the deployment.

* **Enabling Kubernetes in Docker Desktop:**
    * This was a prerequisite. The user confirmed they have Docker Desktop. The steps involve:
        1.  Opening Docker Desktop Settings.
        2.  Navigating to the "Kubernetes" section.
        3.  Checking "Enable Kubernetes".
        4.  Waiting for Docker Desktop to download components and start the single-node cluster.
        5.  Verifying `kubectl config current-context` shows `docker-desktop`.
* **Making Docker Image Accessible:**
    * For Docker Desktop Kubernetes, locally built images (like `hvs-prediction-api:latest`) are typically directly accessible by the Kubernetes cluster it manages. No separate image push to a registry is usually needed for this local setup if `imagePullPolicy` is set correctly.
* **Updating Kubernetes Manifests (`k8s/` directory):**
    * **`k8s/deployment.yaml` (ID: `k8s_deployment_v2_volumemount`):** This file defines the desired state for our API application deployment.
        * **`apiVersion: apps/v1`, `kind: Deployment`**: Standard Kubernetes object headers.
        * **`metadata.name: hvs-api-deployment`**: Name of our deployment.
        * **`spec.replicas: 3`**: Specifies that we want Kubernetes to run **3 identical instances (Pods)** of our API container. This provides basic load distribution and fault tolerance.
        * **`spec.selector.matchLabels: {app: hvs-api}`**: Tells the Deployment which Pods it manages.
        * **`spec.template.metadata.labels: {app: hvs-api}`**: Ensures Pods created by this Deployment get this label.
        * **`spec.template.spec.containers[]`**: Defines the container to run.
            * `name: hvs-api-container`
            * `image: hvs-prediction-api:latest`: The Docker image to use.
            * `imagePullPolicy: IfNotPresent`: Important for local development with Docker Desktop. Kubernetes will use the local image if it exists and won't try to pull from an external registry unless it's not found locally.
            * `ports: [{containerPort: 5001}]`: The port the Flask app listens on inside the container.
            * `env: [{name: BEST_MLFLOW_RUN_ID, value: "your_run_id"}]`: Sets the environment variable for the MLflow Run ID. This must be set to the correct Run ID of the model to load.
            * **`volumeMounts` & `volumes` (Crucial Fix):**
                ```yaml
                volumeMounts:
                - name: mlflow-data-volume
                  mountPath: /app/mlruns # Where mlruns will be inside the container
                volumes:
                - name: mlflow-data-volume
                  hostPath:
                    # Path on the Windows host machine, mapped by Docker Desktop
                    path: /run/desktop/mnt/host/d/Machine_learning/Projects/hvs_fire_prediction/mlruns 
                    type: DirectoryOrCreate
                ```
                This was the key fix to allow the API inside the Kubernetes Pods to access the `mlruns` directory from the host machine. The `hostPath` points to the location of `mlruns` on your Windows D: drive as seen by Docker Desktop's Kubernetes. The `mountPath: /app/mlruns` makes it available at the location where the `MLFLOW_TRACKING_URI` (set in the Dockerfile) expects it.
    * **`k8s/service.yaml` (ID: `k8s_service_v1`):** This file defines how to expose the API pods as a network service.
        * **`apiVersion: v1`, `kind: Service`**: Standard headers.
        * **`metadata.name: hvs-api-service`**: Name of the service.
        * **`spec.selector: {app: hvs-api}`**: This service routes traffic to any Pod with the label `app: hvs-api` (i.e., the Pods created by our Deployment).
        * **`spec.ports[]`**:
            * `name: http-api` (Added for clarity and use in ServiceMonitor later)
            * `port: 80`: The port the Service listens on *inside the cluster*. Other pods in the cluster could reach this service at `http://hvs-api-service:80`.
            * `targetPort: 5001`: The port on the Pods/containers that traffic should be forwarded to (matching the Flask app's port).
        * **`spec.type: NodePort`**: This exposes the Service on a static port on the IP address of each node in the cluster. For Docker Desktop, the node is your local machine (`localhost`). Kubernetes assigns a port from the 30000-32767 range.
* **Applying the Manifests:**
    * **Command:** `kubectl apply -f k8s/service.yaml`
    * **Command:** `kubectl apply -f k8s/deployment.yaml`
    * This tells Kubernetes to create or update these resources to match the desired state defined in the YAML files.
* **Verifying the Deployment:**
    * **Command:** `kubectl get deployments` (to check if replicas are ready).
    * **Command:** `kubectl get pods -l app=hvs-api -w` (to watch pod status; should be `Running` and `1/1 READY`).
    * **Command:** `kubectl logs <pod-name>` (to check application logs inside a specific pod, verifying model/scaler loading and Flask app startup).
    * **Command:** `kubectl get service hvs-api-service` (to find the assigned `NodePort`).
* **Testing the API on Kubernetes:**
    * The API was accessed using `http://localhost:<NodePort>/predict` (where `<NodePort>` was found from `kubectl get svc`).
    * `Invoke-RestMethod` with `test_payload.json` and `test_payload_hazardous.json` was used to send POST requests.
    * Successful JSON responses (including `hazard_prediction: 1` for the hazardous payload) confirmed the API was working correctly on Kubernetes.

**4. Technical Insights & Decisions:**
* **Kubernetes Objects (`Deployment`, `Service`):** Understanding the role of these fundamental Kubernetes objects is key. Deployments manage stateless application replicas, while Services provide stable network access to them.
* **`hostPath` Volumes (for Local Dev):** While `hostPath` is convenient for making local directories (like `mlruns`) available to pods in a local single-node cluster (Docker Desktop, Minikube), it's generally **not recommended for production** in multi-node clusters because it relies on the path existing on the specific node a pod is scheduled to. For production, solutions like PersistentVolumeClaims (PVCs) with network storage, or a centralized MLflow Tracking Server with cloud artifact storage, would be used.
* **`imagePullPolicy: IfNotPresent`:** Important for local development with images built on the same machine where Docker Desktop Kubernetes is running, to avoid unnecessary attempts to pull from remote registries.
* **Labels and Selectors:** The core mechanism by which Kubernetes links Services to Deployments (and Deployments to Pods) is through labels and selectors. The `app: hvs-api` label was consistently used.
* **NodePort Service Type:** Suitable for exposing an application during development on a local Kubernetes cluster. For external access in cloud environments, `LoadBalancer` type services are more common.
* **Debugging `kubectl`:** We learned that the `kubectl describe pod <name>` and `kubectl logs <name>` commands are essential for troubleshooting pod startup issues. The `NameError` encountered highlighted that even with a running K8s deployment, application-level bugs can prevent successful operation, and pod logs are key to finding them.

**5. Key Commands:**
* **Enable K8s in Docker Desktop:** Via Docker Desktop UI settings.
* **Check `kubectl` context:** `kubectl config current-context`
* **Apply manifests:**
    ```powershell
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/deployment.yaml
    ```
* **Verify deployment:**
    ```powershell
    kubectl get deployments
    kubectl get pods -l app=hvs-api
    kubectl describe pod <pod-name> # For troubleshooting
    kubectl logs <pod-name>
    kubectl get service hvs-api-service
    ```
* **Test API (example):**
    ```powershell
    # (After finding NodePort, e.g., 30418)
    Invoke-RestMethod -Uri http://localhost:30418/predict -Method Post -Body (Get-Content -Raw -Path .\test_payload.json) -ContentType "application/json"
    ```

**6. Outcome of Phase 8:**
* The containerized HVS prediction API (`hvs-prediction-api:latest`) successfully deployed to the local Kubernetes cluster provided by Docker Desktop.
* The deployment runs 3 replicas of the API, as specified.
* The API is accessible via a Kubernetes `NodePort` Service.
* Successful end-to-end prediction tests were performed against the API running on Kubernetes, confirming that it loads the MLflow model and scaler (via the `hostPath` volume mount for `mlruns`) and processes requests correctly.
* This phase validated the ability to orchestrate the containerized application using Kubernetes, achieving a key milestone in the MLOps pipeline.

This provides a solid foundation for understanding how the application would behave in a more distributed, orchestrated environment.

---