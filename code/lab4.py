from py2neo import Graph
 
# Connect to the local Neo4j instance
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "neo4jtest"))
 
def query_graph():
    query = """
    MATCH (p:Person)-[:WORKED_WITH]->(p2:Person)
    RETURN p.name AS person, p.profession AS profession, p2.name AS collaborator
    """
    result = graph.run(query).data()
    return result
 
result = query_graph()
print(result)

for record in result:
    print(f"{record['person']} is a {record['profession']} who worked with {record['collaborator']}.\n")
