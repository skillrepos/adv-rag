// ═══════════════════════════════════════════════════════════════════
// OmniTech Knowledge Graph
// Entities and relationships extracted from OmniTech documentation
// ═══════════════════════════════════════════════════════════════════

// ─── DOCUMENTS ───
CREATE (doc1:Document {name: 'Returns_Policy', title: 'Consumer Returns & Refund Policy', version: '2.4', effective_date: '2024-01-01'})
CREATE (doc2:Document {name: 'Troubleshooting_Manual', title: 'Device Troubleshooting Manual', category: 'Technical Support'})
CREATE (doc3:Document {name: 'Security_Handbook', title: 'Account Security Handbook', category: 'Security'})
CREATE (doc4:Document {name: 'Shipping_Guide', title: 'Global Shipping Logistics', category: 'Operations'})

// ─── PRODUCTS ───
CREATE (prod1:Product {name: 'Pro_Series', description: 'Enterprise-grade servers, networking switches, storage arrays', category: 'Enterprise'})
CREATE (prod2:Product {name: 'Standard_Items', description: 'Regular consumer electronics and accessories', category: 'Consumer'})
CREATE (prod3:Product {name: 'Software', description: 'Digital software products and licenses', category: 'Digital'})
CREATE (prod4:Product {name: 'Accessories', description: 'Cables, cases, peripherals', category: 'Consumer'})

// ─── POLICIES ───
CREATE (pol1:Policy {name: 'Standard_Return', description: '30-day return window for standard items', type: 'Return'})
CREATE (pol2:Policy {name: 'Pro_Series_Return', description: '14-day return window with restocking fee', type: 'Return'})
CREATE (pol3:Policy {name: 'DOA_Policy', description: 'Defective on Arrival - full refund, free return shipping', type: 'Warranty'})
CREATE (pol4:Policy {name: 'Holiday_Return', description: 'Extended return window for holiday purchases', type: 'Return'})
CREATE (pol5:Policy {name: 'Software_Policy', description: 'No returns once license activated', type: 'Return'})

// ─── TIME FRAMES ───
CREATE (tf1:TimeFrame {name: '30_Days', value: '30 days', description: 'Standard return window'})
CREATE (tf2:TimeFrame {name: '14_Days', value: '14 days', description: 'Pro-Series return window'})
CREATE (tf3:TimeFrame {name: '7_Days', value: '7 days', description: 'DOA reporting window'})
CREATE (tf4:TimeFrame {name: '5_7_Business_Days', value: '5-7 business days', description: 'Refund processing time'})
CREATE (tf5:TimeFrame {name: 'January_31', value: 'Until January 31st', description: 'Holiday return deadline'})
CREATE (tf6:TimeFrame {name: '48_Hours', value: '48 hours', description: 'Warehouse inspection time'})

// ─── CONTACTS ───
CREATE (con1:Contact {name: 'Returns_Email', value: 'returns@omnitech.example.com', type: 'Email', department: 'Returns'})
CREATE (con2:Contact {name: 'Returns_Phone', value: '1-800-555-0199', hours: 'Mon-Fri 8AM-8PM EST', type: 'Phone', department: 'Returns'})
CREATE (con3:Contact {name: 'Tech_Support', value: 'support@omnitech.example.com', type: 'Email', department: 'Technical Support'})
CREATE (con4:Contact {name: 'International_Support', value: 'international-support@omnitech.example.com', type: 'Email', department: 'International'})

// ─── CONDITIONS ───
CREATE (cond1:Condition {name: 'Like_New', description: 'Original packaging, no signs of use or installation'})
CREATE (cond2:Condition {name: 'Defective', description: 'Dead on arrival or manufacturer defect within warranty'})
CREATE (cond3:Condition {name: 'Used', description: 'Shows evidence of use, installation, or modification'})
CREATE (cond4:Condition {name: 'Damaged', description: 'Physical damage from customer mishandling'})

// ─── FEES ───
CREATE (fee1:Fee {name: 'Standard_Restocking', value: '0%', description: 'No fee for Like-New returns'})
CREATE (fee2:Fee {name: 'Used_Restocking', value: '15-35%', description: 'Fee based on condition assessment'})
CREATE (fee3:Fee {name: 'Pro_Series_Restocking', value: '15%', description: 'Minimum restocking fee for Pro-Series'})
CREATE (fee4:Fee {name: 'Return_Shipping_Small', value: '$8.99', description: 'Items under 5 lbs'})
CREATE (fee5:Fee {name: 'Return_Shipping_Medium', value: '$15.99', description: 'Items 5-20 lbs'})

// ─── SHIPPING METHODS ───
CREATE (ship1:ShippingMethod {name: 'FedEx_Ground', delivery_time: '3-5 business days', region: 'US'})
CREATE (ship2:ShippingMethod {name: 'UPS_Ground', delivery_time: '3-5 business days', region: 'US'})
CREATE (ship3:ShippingMethod {name: 'DHL_International', delivery_time: '7-14 business days', region: 'International'})
CREATE (ship4:ShippingMethod {name: 'USPS_APO_FPO', delivery_time: '10-21 business days', region: 'Military'})

// ═══════════════════════════════════════════════════════════════════
// RELATIONSHIPS
// ═══════════════════════════════════════════════════════════════════

// Document contains policies
CREATE (doc1)-[:CONTAINS]->(pol1)
CREATE (doc1)-[:CONTAINS]->(pol2)
CREATE (doc1)-[:CONTAINS]->(pol3)
CREATE (doc1)-[:CONTAINS]->(pol4)
CREATE (doc1)-[:CONTAINS]->(pol5)

// Policies apply to products
CREATE (pol1)-[:APPLIES_TO]->(prod2)
CREATE (pol1)-[:APPLIES_TO]->(prod4)
CREATE (pol2)-[:APPLIES_TO]->(prod1)
CREATE (pol3)-[:APPLIES_TO]->(prod1)
CREATE (pol3)-[:APPLIES_TO]->(prod2)
CREATE (pol5)-[:APPLIES_TO]->(prod3)

// Policies have time frames
CREATE (pol1)-[:HAS_TIMEFRAME]->(tf1)
CREATE (pol2)-[:HAS_TIMEFRAME]->(tf2)
CREATE (pol3)-[:HAS_TIMEFRAME]->(tf3)
CREATE (pol4)-[:HAS_TIMEFRAME]->(tf5)
CREATE (pol1)-[:REFUND_PROCESSED_IN]->(tf4)
CREATE (pol2)-[:REFUND_PROCESSED_IN]->(tf4)

// Policies require conditions
CREATE (pol1)-[:REQUIRES_CONDITION]->(cond1)
CREATE (pol2)-[:REQUIRES_CONDITION]->(cond1)
CREATE (pol3)-[:REQUIRES_CONDITION]->(cond2)

// Conditions incur fees
CREATE (cond1)-[:INCURS_FEE]->(fee1)
CREATE (cond3)-[:INCURS_FEE]->(fee2)
CREATE (prod1)-[:INCURS_FEE]->(fee3)

// Contacts handle policies/documents
CREATE (con1)-[:HANDLES]->(pol1)
CREATE (con1)-[:HANDLES]->(pol2)
CREATE (con2)-[:HANDLES]->(pol1)
CREATE (con2)-[:HANDLES]->(pol2)
CREATE (con3)-[:HANDLES]->(pol3)
CREATE (con3)-[:HANDLES]->(doc2)
CREATE (con4)-[:HANDLES]->(doc4)

// Documents cover products
CREATE (doc2)-[:COVERS]->(prod1)
CREATE (doc2)-[:COVERS]->(prod2)
CREATE (doc3)-[:COVERS]->(prod3)

// Shipping methods used for returns
CREATE (pol1)-[:USES_SHIPPING]->(ship1)
CREATE (pol1)-[:USES_SHIPPING]->(ship2)
CREATE (pol3)-[:USES_SHIPPING]->(ship1)
CREATE (doc4)-[:DESCRIBES]->(ship3)
CREATE (doc4)-[:DESCRIBES]->(ship4)

// Return shipping fees
CREATE (pol1)-[:HAS_SHIPPING_FEE]->(fee4)
CREATE (pol1)-[:HAS_SHIPPING_FEE]->(fee5)

// ─── INDEXES FOR PERFORMANCE ───
;
