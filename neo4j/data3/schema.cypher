LOAD CSV WITH HEADERS FROM 'file:///omnitech_policies.csv' AS row

FOREACH (_ in CASE WHEN row.policy_name IS NOT NULL AND row.policy_name <> '' THEN [1] ELSE [] END |
    MERGE (pol:Policy {name: row.policy_name})
    SET pol.description = row.policy_description,
        pol.type = row.policy_type)

FOREACH (_ in CASE WHEN row.product_name IS NOT NULL AND row.product_name <> '' THEN [1] ELSE [] END |
    MERGE (prod:Product {name: row.product_name})
    SET prod.description = row.product_description,
        prod.category = row.product_category)

FOREACH (_ in CASE WHEN row.timeframe_name IS NOT NULL AND row.timeframe_name <> '' THEN [1] ELSE [] END |
    MERGE (tf:TimeFrame {name: row.timeframe_name})
    SET tf.value = row.timeframe_value,
        tf.description = row.timeframe_desc)

FOREACH (_ in CASE WHEN row.contact_name IS NOT NULL AND row.contact_name <> '' THEN [1] ELSE [] END |
    MERGE (con:Contact {name: row.contact_name})
    SET con.value = row.contact_value,
        con.type = row.contact_type,
        con.department = row.contact_department,
        con.hours = row.contact_hours)

FOREACH (_ in CASE WHEN row.condition_name IS NOT NULL AND row.condition_name <> '' THEN [1] ELSE [] END |
    MERGE (cond:Condition {name: row.condition_name})
    SET cond.description = row.condition_description)

FOREACH (_ in CASE WHEN row.fee_name IS NOT NULL AND row.fee_name <> '' THEN [1] ELSE [] END |
    MERGE (fee:Fee {name: row.fee_name})
    SET fee.value = row.fee_value,
        fee.description = row.fee_description)

FOREACH (_ in CASE WHEN row.shipping_name IS NOT NULL AND row.shipping_name <> '' THEN [1] ELSE [] END |
    MERGE (ship:ShippingMethod {name: row.shipping_name})
    SET ship.delivery_time = row.shipping_time,
        ship.region = row.shipping_region)

FOREACH (_ in CASE WHEN row.document_name IS NOT NULL AND row.document_name <> '' THEN [1] ELSE [] END |
    MERGE (doc:Document {name: row.document_name})
    SET doc.title = row.document_title,
        doc.category = row.document_category)

WITH row
WHERE row.policy_name IS NOT NULL AND row.policy_name <> ''
MATCH (pol:Policy {name: row.policy_name})

FOREACH (_ in CASE WHEN row.product_name IS NOT NULL AND row.product_name <> '' THEN [1] ELSE [] END |
    MERGE (prod:Product {name: row.product_name})
    MERGE (pol)-[:APPLIES_TO]->(prod))

FOREACH (_ in CASE WHEN row.timeframe_name IS NOT NULL AND row.timeframe_name <> '' THEN [1] ELSE [] END |
    MERGE (tf:TimeFrame {name: row.timeframe_name})
    MERGE (pol)-[:HAS_TIMEFRAME]->(tf))

FOREACH (_ in CASE WHEN row.contact_name IS NOT NULL AND row.contact_name <> '' THEN [1] ELSE [] END |
    MERGE (con:Contact {name: row.contact_name})
    MERGE (con)-[:HANDLES]->(pol))

FOREACH (_ in CASE WHEN row.condition_name IS NOT NULL AND row.condition_name <> '' THEN [1] ELSE [] END |
    MERGE (cond:Condition {name: row.condition_name})
    MERGE (pol)-[:REQUIRES_CONDITION]->(cond))

FOREACH (_ in CASE WHEN row.fee_name IS NOT NULL AND row.fee_name <> '' THEN [1] ELSE [] END |
    MERGE (fee:Fee {name: row.fee_name})
    MERGE (pol)-[:HAS_FEE]->(fee))

FOREACH (_ in CASE WHEN row.shipping_name IS NOT NULL AND row.shipping_name <> '' THEN [1] ELSE [] END |
    MERGE (ship:ShippingMethod {name: row.shipping_name})
    MERGE (pol)-[:USES_SHIPPING]->(ship))

FOREACH (_ in CASE WHEN row.document_name IS NOT NULL AND row.document_name <> '' THEN [1] ELSE [] END |
    MERGE (doc:Document {name: row.document_name})
    MERGE (doc)-[:CONTAINS]->(pol))

