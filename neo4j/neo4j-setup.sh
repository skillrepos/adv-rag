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

echo "=========================================="
echo "Cleaning up previous Neo4j instance..."
echo "=========================================="

# Stop and remove container
docker stop "$CONTAINER_NAME" 2>/dev/null
docker rm -f "$CONTAINER_NAME" 2>/dev/null

# Remove the named volume to clear old data
docker volume rm neo4j_data 2>/dev/null

# Remove the image to force complete rebuild
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

# Create fresh volume
docker volume create neo4j_data >/dev/null

# Start container (suppress container ID output)
docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -v neo4j_data:/data \
     --env NEO4J_AUTH=neo4j/neo4jtest \
     --env NEO4J_PLUGINS='["apoc"]' \
     --env NEO4J_apoc_import_file_enabled=true \
     --env NEO4J_apoc_import_file_use__neo4j__config=true \
     neo4j:custom >/dev/null

echo ""
echo "Waiting for Neo4j to start (this takes 30-60 seconds)..."

# Wait for Neo4j to be ready
READY=false
for i in {1..60}; do
    if docker exec neo4j cypher-shell -u neo4j -p neo4jtest "RETURN 1" >/dev/null 2>&1; then
        READY=true
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

if [ "$READY" = false ]; then
    echo "ERROR: Neo4j failed to start within 2 minutes."
    echo "Check logs with: docker logs neo4j"
    exit 1
fi

echo "Neo4j is ready!"

# Wait for APOC plugin to initialize
echo "Waiting for APOC to initialize..."
sleep 10

# Check node count
COUNT=$(docker exec neo4j cypher-shell -u neo4j -p neo4jtest "MATCH (n) RETURN count(n) as c" 2>/dev/null | grep -E "^[0-9]+$" | head -1)

# If no nodes loaded, try manual load
if [ -z "$COUNT" ] || [ "$COUNT" = "0" ]; then
    echo "APOC auto-init didn't load schema. Loading manually..."
    docker exec neo4j cypher-shell -u neo4j -p neo4jtest -f /var/lib/neo4j/db_init/schema.cypher 2>/dev/null
    sleep 2
    COUNT=$(docker exec neo4j cypher-shell -u neo4j -p neo4jtest "MATCH (n) RETURN count(n) as c" 2>/dev/null | grep -E "^[0-9]+$" | head -1)
fi

echo ""
echo "=========================================="
echo "Loaded ${COUNT:-0} nodes into the graph."
echo "=========================================="

# Check if running in Codespaces
if [ -n "$CODESPACES" ]; then
    echo ""
    echo "GITHUB CODESPACES DETECTED"
    echo "=========================================="
    echo "To access Neo4j Browser:"
    echo "  1. Go to the PORTS tab in VS Code"
    echo "  2. Click 'Add Port' and enter 7474"
    echo "  3. Click the globe icon to open in browser"
    echo ""
    echo "Python will use: neo4j://localhost:7687"
    echo "=========================================="
else
    echo ""
    echo "Web interface: http://localhost:7474"
    echo "Bolt URI:      neo4j://localhost:7687"
fi
echo "Login:         neo4j / neo4jtest"
echo ""
echo "To view logs:  docker logs neo4j"
echo "To stop:       docker stop neo4j"
echo "=========================================="


