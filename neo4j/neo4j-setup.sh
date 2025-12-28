#!/bin/bash
CONTAINER_NAME="neo4j"
IMAGE_NAME="neo4j:custom"
DATASET="$1"

if [ -z "$DATASET" ]; then
    echo "Usage: ./neo4j-setup.sh <dataset_number>"
    echo "  1 = Simple person graph (Lab 4)"
    echo "  2 = Movie database (Lab 5)"
    echo "  3 = OmniTech knowledge graph (Lab 6)"
    exit 1
fi

# Force remove existing container (if any)
echo "Cleaning up existing container..."
docker stop "$CONTAINER_NAME" 2>/dev/null
docker rm -f "$CONTAINER_NAME" 2>/dev/null

# Force remove existing image (if any)
echo "Cleaning up existing image..."
docker rmi -f "$IMAGE_NAME" 2>/dev/null

echo ""
echo "=========================================="
echo "Building Neo4j image with dataset $DATASET..."
echo "=========================================="
docker build --no-cache -f Dockerfile_data$DATASET -t neo4j:custom .

echo ""
echo "=========================================="
echo "Starting Neo4j container..."
echo "=========================================="
docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     --env NEO4J_AUTH=neo4j/neo4jtest \
     --env NEO4J_PLUGINS='["apoc"]' \
     --env NEO4J_apoc_import_file_enabled=true \
     --env NEO4J_apoc_import_file_use__neo4j__config=true \
     neo4j:custom

echo ""
echo "Waiting for Neo4j to start (this takes 30-60 seconds)..."
echo ""

# Wait for Neo4j to be ready
READY=false
for i in {1..60}; do
    if docker exec neo4j cypher-shell -u neo4j -p neo4jtest "RETURN 1" > /dev/null 2>&1; then
        READY=true
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

if [ "$READY" = true ]; then
    echo "Neo4j is ready!"

    # Give APOC initializer time to run
    echo "Waiting for APOC to initialize schema..."
    sleep 5

    # Verify node count
    COUNT=$(docker exec neo4j cypher-shell -u neo4j -p neo4jtest "MATCH (n) RETURN count(n)" 2>/dev/null | tail -1)
    echo "Loaded $COUNT nodes into the graph."

    echo ""
    echo "=========================================="
    echo "Neo4j is running!"
    echo "=========================================="
    echo "Web interface: http://localhost:7474"
    echo "Bolt URI:      neo4j://localhost:7687"
    echo "Login:         neo4j / neo4jtest"
    echo ""
    echo "To view logs:  docker logs neo4j"
    echo "To stop:       docker stop neo4j"
    echo "=========================================="
else
    echo "ERROR: Neo4j failed to start within 2 minutes."
    echo "Check logs with: docker logs neo4j"
    exit 1
fi

