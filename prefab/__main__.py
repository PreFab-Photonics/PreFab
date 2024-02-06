"""Main entry point for the Prefab CLI."""
import argparse
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import toml


def store_jwt_securely(jwt, refresh_token):
    """
    Store the JWT and refresh token securely in a TOML file.
    """
    prefab_file_path = os.path.expanduser("~/.prefab.toml")
    with open(prefab_file_path, "w", encoding="utf-8") as toml_file:
        toml.dump({"access_token": jwt, "refresh_token": refresh_token}, toml_file)
    print(f"Token successfully stored in {prefab_file_path}")


class GracefulHTTPServer(HTTPServer):
    """An HTTPServer that supports graceful shutdown."""

    def shutdown(self):
        """Stop the serve_forever loop."""
        self._BaseServer__shutdown_request = True
        self.server_close()


class CallbackHandler(BaseHTTPRequestHandler):
    """
    A request handler for the HTTP server that handles the OAuth callback.
    """

    def do_GET(self):
        if self.path.startswith("/callback"):
            query_params = self.path.split("?")[1]
            params = {
                param.split("=")[0]: param.split("=")[1]
                for param in query_params.split("&")
            }
            jwt_token = params.get("token")
            refresh_token = params.get("refresh_token")
            if jwt_token and refresh_token:
                print("Token verified!")
                store_jwt_securely(jwt_token, refresh_token)
                self.send_response_only(200, "OK")
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"Authentication successful, you can close this window."
                )
                threading.Thread(target=self.server.shutdown).start()
            else:
                self.send_error(400, "Bad Request: Missing tokens in callback URL.")


def main():
    parser = argparse.ArgumentParser(description="Prefab CLI")
    parser.add_argument("command", help="The command to run", choices=["setup"])
    parser.add_argument(
        "--port", help="Port number for the HTTP server", type=int, default=8000
    )

    args = parser.parse_args()

    if args.command == "setup":
        webbrowser.open("https://www.prefabphotonics.com/token-flow")
        httpd = GracefulHTTPServer(("localhost", args.port), CallbackHandler)
        print("Started token authentication flow on the web browser...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        httpd.server_close()
    else:
        print(f"Command {args.command} not recognized.")


if __name__ == "__main__":
    main()
