This program is a web-based traffic analysis tool built with Flask that allows users to upload PCAP files or provide manual input for threat detection. It combines machine learning (XGBoost) and deep learning (CNN) models to classify network traffic as benign or malicious.
- PCAP Parsing: Uses Scapy to read packet captures and extract flow-level features (duration, packet counts, byte rates, window sizes, etc.).
- Feature-Based Detection (XGBoost): Flow statistics are fed into an XGBoost model to predict threat probability.
- Payload-Based Detection (CNN): A simple 1D CNN analyzes raw packet payload bytes to detect malicious signatures.
- Threat Scoring: Results from both models are combined into a final threat level (Low, Medium, High).
- API Endpoint: /analyze accepts file uploads or manual text data and returns JSON with threat level and details.
- Frontend Integration: Serves an index.html page at the root for user interaction.
