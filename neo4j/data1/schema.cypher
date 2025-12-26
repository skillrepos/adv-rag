   CREATE (p:Person {name: 'Ada Lovelace', profession: 'Mathematician'})
   CREATE (p2:Person {name: 'Alan Turing', profession: 'Computer Scientist'})
   CREATE (p)-[:WORKED_WITH]->(p2)
