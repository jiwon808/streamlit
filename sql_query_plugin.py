import requests
from open_webui import WebUIPlugin

class SQLQueryPlugin(WebUIPlugin):
    def on_message(self, message):
        if message.startswith('/sql'):
            query = message[5:].strip()
            response = requests.post(
                'http://localhost:5000/execute_sql', 
                json={'query': query}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Failed to execute query"}

plugin = SQLQueryPlugin()
plugin.run()
