# UNINwalk BE

This repository includes the backend for the **UNINwalk** project.

## Running Locally (Docker)

For every new version of code, rebuild the Docker image:

```bash
docker build -t fastapi-app .
```

Then run the container:

```bash
docker run -p 8000:8000 fastapi-app
```

The API will be available at `http://localhost:8000`.

## Running on the Virtual Machine

1. Connect via **Commands → Open in Putty**
2. Navigate to the project directory:
   ```bash
   cd /var/www/uninwalk-be
   ```
3. Pull the latest changes if there are any:
   ```bash
   git pull
   ```
4. Check what is currently running:
   ```bash
   docker ps
   ```
5. Stop the running container if needed:
   ```bash
   docker stop <container_name>
   ```
6. Build the Docker image and run it in the background:
   ```bash
   docker build -t uninwalk-be .
   docker run -p 8000:8000 -d uninwalk-be
   ```
