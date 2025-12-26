CONTAINER_NAME="neo4j"
IMAGE_NAME="neo4j:custom"

# Check if the container exists
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo "Stopping and removing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
else
    echo "Container $CONTAINER_NAME not found."
fi

# Check if the image exists
if docker images | grep -q "$IMAGE_NAME"; then
    echo "Removing image: $IMAGE_NAME"
    docker rmi "$IMAGE_NAME"
else
    echo "Image $IMAGE_NAME not found."
fi 

docker build -f Dockerfile_data$1 -t neo4j:custom .
docker run \
     --name neo4j \
     -p7474:7474 -p7687:7687 \
     --env NEO4J_dbms_directories_import="/" \
     --env NEO4JLABS_PLUGINS='["apoc"]' \
     --env NEO4J_AUTH=neo4j/neo4jtest \
     neo4j:custom
