from py2neo import Graph
from pyvis.network import Network

# Connect to your Neo4j database
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "neo4jtest"))

# Execute a Cypher query to retrieve data
query = """
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
RETURN p.name AS person_name, m.title AS movie_title
"""
result = graph.run(query).data()

# Create a Pyvis network object
net = Network(notebook=True)

# Add nodes and edges to the network
for record in result:
    net.add_node(record["person_name"], label=record["person_name"])
    net.add_node(record["movie_title"], label=record["movie_title"])
    net.add_edge(record["person_name"], record["movie_title"])

query = """
MATCH (m:Movie)-[r:IN_GENRE]->(g:Genre)
RETURN m.title AS movie_title, g.name AS movie_genre
"""
result = graph.run(query).data()

# Add nodes and edges to the network
for record in result:
    net.add_node(record["movie_genre"], label=record["movie_genre"])
    net.add_node(record["movie_title"], label=record["movie_title"])
    net.add_edge(record["movie_title"], record["movie_genre"])

# Visualize the network
net.save_graph("neo4j_graph.html")
