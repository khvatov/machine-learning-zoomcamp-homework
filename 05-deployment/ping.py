from typing import Literal
from flask import Flask


if __name__ == "__main__":
    app = Flask(__name__)

    @app.route("/ping", methods=["GET"])
    def ping()->Literal["PONG"]:
        return "PONG"
    
    app.run(debug=True, host="0.0.0.0", port=9696)  # run the app