import sys

sys.path.insert(0, "/home/user/stitchpro-backend")
from api.app import app

app.run(host="0.0.0.0", port=5050, debug=False)
