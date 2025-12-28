// ═══════════════════════════════════════════════════════════════════
// OmniTech Knowledge Graph
// Entities and relationships extracted from OmniTech documentation
// ═══════════════════════════════════════════════════════════════════

// ─── DOCUMENTS ───
CREATE (:Document {name: 'Returns_Policy', title: 'Consumer Returns & Refund Policy', version: '2.4', effective_date: '2024-01-01'});
CREATE (:Document {name: 'Troubleshooting_Manual', title: 'Device Troubleshooting Manual', category: 'Technical Support'});
CREATE (:Document {name: 'Security_Handbook', title: 'Account Security Handbook', category: 'Security'});
CREATE (:Document {name: 'Shipping_Guide', title: 'Global Shipping Logistics', category: 'Operations'});

// ─── PRODUCTS ───
CREATE (:Product {name: 'Pro_Series', description: 'Enterprise-grade servers, networking switches, storage arrays', category: 'Enterprise'});
CREATE (:Product {name: 'Standard_Items', description: 'Regular consumer electronics and accessories', category: 'Consumer'});
CREATE (:Product {name: 'Software', description: 'Digital software products and licenses', category: 'Digital'});
CREATE (:Product {name: 'Accessories', description: 'Cables, cases, peripherals', category: 'Consumer'});

// ─── POLICIES ───
CREATE (:Policy {name: 'Standard_Return', description: '30-day return window for standard items', type: 'Return'});
CREATE (:Policy {name: 'Pro_Series_Return', description: '14-day return window with restocking fee', type: 'Return'});
CREATE (:Policy {name: 'DOA_Policy', description: 'Defective on Arrival - full refund, free return shipping', type: 'Warranty'});
CREATE (:Policy {name: 'Holiday_Return', description: 'Extended return window for holiday purchases', type: 'Return'});
CREATE (:Policy {name: 'Software_Policy', description: 'No returns once license activated', type: 'Return'});

// ─── TIME FRAMES ───
CREATE (:TimeFrame {name: '30_Days', value: '30 days', description: 'Standard return window'});
CREATE (:TimeFrame {name: '14_Days', value: '14 days', description: 'Pro-Series return window'});
CREATE (:TimeFrame {name: '7_Days', value: '7 days', description: 'DOA reporting window'});
CREATE (:TimeFrame {name: '5_7_Business_Days', value: '5-7 business days', description: 'Refund processing time'});
CREATE (:TimeFrame {name: 'January_31', value: 'Until January 31st', description: 'Holiday return deadline'});
CREATE (:TimeFrame {name: '48_Hours', value: '48 hours', description: 'Warehouse inspection time'});

// ─── CONTACTS ───
CREATE (:Contact {name: 'Returns_Email', value: 'returns@omnitech.example.com', type: 'Email', department: 'Returns'});
CREATE (:Contact {name: 'Returns_Phone', value: '1-800-555-0199', hours: 'Mon-Fri 8AM-8PM EST', type: 'Phone', department: 'Returns'});
CREATE (:Contact {name: 'Tech_Support', value: 'support@omnitech.example.com', type: 'Email', department: 'Technical Support'});
CREATE (:Contact {name: 'International_Support', value: 'international-support@omnitech.example.com', type: 'Email', department: 'International'});

// ─── CONDITIONS ───
CREATE (:Condition {name: 'Like_New', description: 'Original packaging, no signs of use or installation'});
CREATE (:Condition {name: 'Defective', description: 'Dead on arrival or manufacturer defect within warranty'});
CREATE (:Condition {name: 'Used', description: 'Shows evidence of use, installation, or modification'});
CREATE (:Condition {name: 'Damaged', description: 'Physical damage from customer mishandling'});

// ─── FEES ───
CREATE (:Fee {name: 'Standard_Restocking', value: '0%', description: 'No fee for Like-New returns'});
CREATE (:Fee {name: 'Used_Restocking', value: '15-35%', description: 'Fee based on condition assessment'});
CREATE (:Fee {name: 'Pro_Series_Restocking', value: '15%', description: 'Minimum restocking fee for Pro-Series'});
CREATE (:Fee {name: 'Return_Shipping_Small', value: '$8.99', description: 'Items under 5 lbs'});
CREATE (:Fee {name: 'Return_Shipping_Medium', value: '$15.99', description: 'Items 5-20 lbs'});

// ─── SHIPPING METHODS ───
CREATE (:ShippingMethod {name: 'FedEx_Ground', delivery_time: '3-5 business days', region: 'US'});
CREATE (:ShippingMethod {name: 'UPS_Ground', delivery_time: '3-5 business days', region: 'US'});
CREATE (:ShippingMethod {name: 'DHL_International', delivery_time: '7-14 business days', region: 'International'});
CREATE (:ShippingMethod {name: 'USPS_APO_FPO', delivery_time: '10-21 business days', region: 'Military'});

// ═══════════════════════════════════════════════════════════════════
// RELATIONSHIPS - Using MATCH to find nodes by name
// ═══════════════════════════════════════════════════════════════════

// Document contains policies
MATCH (doc:Document {name: 'Returns_Policy'}), (pol:Policy {name: 'Standard_Return'}) CREATE (doc)-[:CONTAINS]->(pol);
MATCH (doc:Document {name: 'Returns_Policy'}), (pol:Policy {name: 'Pro_Series_Return'}) CREATE (doc)-[:CONTAINS]->(pol);
MATCH (doc:Document {name: 'Returns_Policy'}), (pol:Policy {name: 'DOA_Policy'}) CREATE (doc)-[:CONTAINS]->(pol);
MATCH (doc:Document {name: 'Returns_Policy'}), (pol:Policy {name: 'Holiday_Return'}) CREATE (doc)-[:CONTAINS]->(pol);
MATCH (doc:Document {name: 'Returns_Policy'}), (pol:Policy {name: 'Software_Policy'}) CREATE (doc)-[:CONTAINS]->(pol);

// Policies apply to products
MATCH (pol:Policy {name: 'Standard_Return'}), (prod:Product {name: 'Standard_Items'}) CREATE (pol)-[:APPLIES_TO]->(prod);
MATCH (pol:Policy {name: 'Standard_Return'}), (prod:Product {name: 'Accessories'}) CREATE (pol)-[:APPLIES_TO]->(prod);
MATCH (pol:Policy {name: 'Pro_Series_Return'}), (prod:Product {name: 'Pro_Series'}) CREATE (pol)-[:APPLIES_TO]->(prod);
MATCH (pol:Policy {name: 'DOA_Policy'}), (prod:Product {name: 'Pro_Series'}) CREATE (pol)-[:APPLIES_TO]->(prod);
MATCH (pol:Policy {name: 'DOA_Policy'}), (prod:Product {name: 'Standard_Items'}) CREATE (pol)-[:APPLIES_TO]->(prod);
MATCH (pol:Policy {name: 'Software_Policy'}), (prod:Product {name: 'Software'}) CREATE (pol)-[:APPLIES_TO]->(prod);

// Policies have time frames
MATCH (pol:Policy {name: 'Standard_Return'}), (tf:TimeFrame {name: '30_Days'}) CREATE (pol)-[:HAS_TIMEFRAME]->(tf);
MATCH (pol:Policy {name: 'Pro_Series_Return'}), (tf:TimeFrame {name: '14_Days'}) CREATE (pol)-[:HAS_TIMEFRAME]->(tf);
MATCH (pol:Policy {name: 'DOA_Policy'}), (tf:TimeFrame {name: '7_Days'}) CREATE (pol)-[:HAS_TIMEFRAME]->(tf);
MATCH (pol:Policy {name: 'Holiday_Return'}), (tf:TimeFrame {name: 'January_31'}) CREATE (pol)-[:HAS_TIMEFRAME]->(tf);
MATCH (pol:Policy {name: 'Standard_Return'}), (tf:TimeFrame {name: '5_7_Business_Days'}) CREATE (pol)-[:REFUND_PROCESSED_IN]->(tf);
MATCH (pol:Policy {name: 'Pro_Series_Return'}), (tf:TimeFrame {name: '5_7_Business_Days'}) CREATE (pol)-[:REFUND_PROCESSED_IN]->(tf);

// Policies require conditions
MATCH (pol:Policy {name: 'Standard_Return'}), (cond:Condition {name: 'Like_New'}) CREATE (pol)-[:REQUIRES_CONDITION]->(cond);
MATCH (pol:Policy {name: 'Pro_Series_Return'}), (cond:Condition {name: 'Like_New'}) CREATE (pol)-[:REQUIRES_CONDITION]->(cond);
MATCH (pol:Policy {name: 'DOA_Policy'}), (cond:Condition {name: 'Defective'}) CREATE (pol)-[:REQUIRES_CONDITION]->(cond);

// Conditions incur fees
MATCH (cond:Condition {name: 'Like_New'}), (fee:Fee {name: 'Standard_Restocking'}) CREATE (cond)-[:INCURS_FEE]->(fee);
MATCH (cond:Condition {name: 'Used'}), (fee:Fee {name: 'Used_Restocking'}) CREATE (cond)-[:INCURS_FEE]->(fee);
MATCH (prod:Product {name: 'Pro_Series'}), (fee:Fee {name: 'Pro_Series_Restocking'}) CREATE (prod)-[:INCURS_FEE]->(fee);

// Contacts handle policies/documents
MATCH (con:Contact {name: 'Returns_Email'}), (pol:Policy {name: 'Standard_Return'}) CREATE (con)-[:HANDLES]->(pol);
MATCH (con:Contact {name: 'Returns_Email'}), (pol:Policy {name: 'Pro_Series_Return'}) CREATE (con)-[:HANDLES]->(pol);
MATCH (con:Contact {name: 'Returns_Phone'}), (pol:Policy {name: 'Standard_Return'}) CREATE (con)-[:HANDLES]->(pol);
MATCH (con:Contact {name: 'Returns_Phone'}), (pol:Policy {name: 'Pro_Series_Return'}) CREATE (con)-[:HANDLES]->(pol);
MATCH (con:Contact {name: 'Tech_Support'}), (pol:Policy {name: 'DOA_Policy'}) CREATE (con)-[:HANDLES]->(pol);
MATCH (con:Contact {name: 'Tech_Support'}), (doc:Document {name: 'Troubleshooting_Manual'}) CREATE (con)-[:HANDLES]->(doc);
MATCH (con:Contact {name: 'International_Support'}), (doc:Document {name: 'Shipping_Guide'}) CREATE (con)-[:HANDLES]->(doc);

// Documents cover products
MATCH (doc:Document {name: 'Troubleshooting_Manual'}), (prod:Product {name: 'Pro_Series'}) CREATE (doc)-[:COVERS]->(prod);
MATCH (doc:Document {name: 'Troubleshooting_Manual'}), (prod:Product {name: 'Standard_Items'}) CREATE (doc)-[:COVERS]->(prod);
MATCH (doc:Document {name: 'Security_Handbook'}), (prod:Product {name: 'Software'}) CREATE (doc)-[:COVERS]->(prod);

// Shipping methods used for returns
MATCH (pol:Policy {name: 'Standard_Return'}), (ship:ShippingMethod {name: 'FedEx_Ground'}) CREATE (pol)-[:USES_SHIPPING]->(ship);
MATCH (pol:Policy {name: 'Standard_Return'}), (ship:ShippingMethod {name: 'UPS_Ground'}) CREATE (pol)-[:USES_SHIPPING]->(ship);
MATCH (pol:Policy {name: 'DOA_Policy'}), (ship:ShippingMethod {name: 'FedEx_Ground'}) CREATE (pol)-[:USES_SHIPPING]->(ship);
MATCH (doc:Document {name: 'Shipping_Guide'}), (ship:ShippingMethod {name: 'DHL_International'}) CREATE (doc)-[:DESCRIBES]->(ship);
MATCH (doc:Document {name: 'Shipping_Guide'}), (ship:ShippingMethod {name: 'USPS_APO_FPO'}) CREATE (doc)-[:DESCRIBES]->(ship);

// Return shipping fees
MATCH (pol:Policy {name: 'Standard_Return'}), (fee:Fee {name: 'Return_Shipping_Small'}) CREATE (pol)-[:HAS_SHIPPING_FEE]->(fee);
MATCH (pol:Policy {name: 'Standard_Return'}), (fee:Fee {name: 'Return_Shipping_Medium'}) CREATE (pol)-[:HAS_SHIPPING_FEE]->(fee);

// ─── INDEXES FOR PERFORMANCE ───
CREATE INDEX product_name IF NOT EXISTS FOR (p:Product) ON (p.name);
CREATE INDEX policy_name IF NOT EXISTS FOR (p:Policy) ON (p.name);
CREATE INDEX contact_name IF NOT EXISTS FOR (c:Contact) ON (c.name);
CREATE INDEX timeframe_name IF NOT EXISTS FOR (t:TimeFrame) ON (t.name);
