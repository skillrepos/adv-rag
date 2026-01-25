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
  docker stop "$CONTAINER_NAME" 2>/dev/null
  docker rm -f "$CONTAINER_NAME" 2>/dev/null
  docker volume rm neo4j_data 2>/dev/null
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
  docker volume create neo4j_data >/dev/null
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
      exit 1
  fi

  echo "Neo4j is ready!"
  echo "Loading schema from /var/lib/neo4j/db_init/schema.cypher..."
  docker exec neo4j cypher-shell -u neo4j -p neo4jtest -f /var/lib/neo4j/db_init/schema.cypher
  echo "Schema loaded!"

  echo ""
  echo "=========================================="
  echo "Neo4j is running with dataset $DATASET!"
  echo "=========================================="
  echo "Neo4j Browser: http://localhost:7474"
  echo "Bolt URI:      neo4j://localhost:7687"
  echo "Login:         neo4j / neo4jtest"
  echo ""
  echo "Verify with: docker exec -it neo4j cypher-shell -u neo4j -p neo4jtest"
  echo "Then run:    MATCH (n) RETURN count(n);"
  echo "=========================================="
