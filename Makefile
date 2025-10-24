.PHONY: install setup train clean

# Variables
PYTHON := python3
PDM := pdm
TRAIN_SCRIPT := train.py

# Docker variables
DOCKER_HUB_USERNAME := prasadzende
DOCKER_IMAGE_NAME := iris-fastapi
DOCKER_IMAGE_TAG := latest
CONTAINER_NAME := iris-api-container
MODEL_FILE := ./src/model_src/iris_model.pkl

# Default target
all: setup install train

# Install PDM
setup:
	@echo "Installing PDM..."
	@curl -sSL https://pdm.fming.dev/install-pdm.py | $(PYTHON) -
	@echo "PDM installation completed"

# Install dependencies using PDM
install:
	@echo "Installing dependencies..."
	@cd ./src/model_src && $(PDM) sync --prod
	@echo "Dependencies installation completed"

# Train the model with default parameters
train:
	@echo "Training model..."
	@cd ./src/model_src && pdm run python $(TRAIN_SCRIPT) \
        --max-iter 200 \
        --solver lbfgs \
        --random-state 42 \
        --test-size 0.2 \
		--mlflow-tracking-uri http://localhost:5005
	@echo "Training completed"

train-no-registration:
	@echo "Training model without MLflow registration..."
	@cd ./src/model_src && pdm run python $(TRAIN_SCRIPT) \
        --max-iter 200 \
        --solver lbfgs \
        --random-state 42 \
        --test-size 0.2 \
		--no-register-model
	@echo "Training completed"

# Copy model to API directory
copy-model:
	@echo "Copying model to api directory..."
	@cp $(MODEL_FILE) ./src/api/
	@echo "Model copied to .src/api/"

# Docker tasks
build-docker:
	@echo "Building docker image..."
	@ls -lrta ./src/api
	@docker build -t $(DOCKER_HUB_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) ./src/api/
	@echo "Docker image built"

push-docker:
	@echo "Pushing docker image to Docker Hub..."
	@docker push $(DOCKER_HUB_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
	@echo "Docker image pushed"

run-docker:
	@echo "Running docker container..."
	@docker run -d --name $(CONTAINER_NAME) -p 8000:8000 $(DOCKER_HUB_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
	@echo "Container $(CONTAINER_NAME) is running on port 8000"

stop-docker:
	@echo "Stopping and removing docker container..."
	@docker stop $(CONTAINER_NAME) || true
	@docker rm $(CONTAINER_NAME) || true
	@echo "Container stopped and removed"

test-endpoint:
	@echo "Waiting for the API to be ready..."
	@sleep 10
	@echo "Sending prediction request to http://localhost:8000/predict"
	@RESPONSE=$$(curl -s -X POST "http://localhost:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{ \
	"sepal_length": 5.1, \
	"sepal_width": 3.5, \
	"petal_length": 1.4, \
	"petal_width": 0.2 \
	}'); \
	EXPECTED='{"status":"success","prediction":"setosa"}'
	echo "Response: $$RESPONSE"

	# Extract status and prediction fields from RESPONSE ignoring probability
	ACTUAL=$(echo "$$RESPONSE" | jq '{status, prediction}' | tr -d '\n')

	if [ "$$ACTUAL" = "$$EXPECTED" ]; then
		echo "✅ Test passed: Prediction is correct."
	else
		echo "❌ Test failed: Unexpected response."
		exit 1
	fi

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -f iris_model.pkl
	@rm -rf __pycache__
	@rm -rf .pdm-build
	@rm -rf dist
	@echo "Cleanup completed"