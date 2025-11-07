"""Provides the main entry point for the Prefab authentication CLI."""

import argparse
import os
import threading
import webbrowser
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer

import toml


def store_jwt(jwt, refresh_token):
    """Store the JWT and refresh token in a TOML file."""
    prefab_file_path = os.path.expanduser("~/.prefab.toml")
    with open(prefab_file_path, "w", encoding="utf-8") as toml_file:
        toml.dump({"access_token": jwt, "refresh_token": refresh_token}, toml_file)
    print(
        f"Token successfully stored in {prefab_file_path}.\n\n"
        "🎉 Welcome to PreFab.\n"
        "See our examples at https://docs.prefabphotonics.com/examples to start.\n"
        "Reach out to us at hi@prefabphotonics.com for support."
    )


class GracefulHTTPServer(HTTPServer):
    """An HTTPServer that supports graceful shutdown."""

    def shutdown(self):
        """Stop the serve_forever loop."""
        self._BaseServer__shutdown_request = True
        self.server_close()


class CallbackHandler(BaseHTTPRequestHandler):
    """A request handler for the HTTP server that handles the JWT-auth callback."""

    def log_message(self, format, *args):
        """Suppress HTTP request logging."""
        pass

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
                print("Token verified.")
                store_jwt(jwt_token, refresh_token)
                self.send_response(302)
                self.send_header(
                    "Location",
                    "https://www.prefabphotonics.com/auth/token-flow/success",
                )
                self.end_headers()
                threading.Thread(target=self.server.shutdown).start()
            else:
                self.send_error(400, "Bad Request: Missing tokens in callback URL.")


def main():
    """Main function for the Prefab authentication CLI."""
    parser = argparse.ArgumentParser(description="PreFab Auth CLI")
    parser.add_argument("command", help="The command to run", choices=["setup"])
    parser.add_argument(
        "--port", help="Port number for the HTTP server", type=int, default=8000
    )
    parser.add_argument(
        "--timeout", help="Timeout in seconds for authentication", type=int, default=300
    )
    args = parser.parse_args()

    if args.command == "setup":
        webbrowser.open("https://www.prefabphotonics.com/auth/token-flow")
        httpd = GracefulHTTPServer(("localhost", args.port), CallbackHandler)
        print("Started token authentication flow on the web browser...")

        def timeout_handler():
            print("\nAuthentication timed out. Please run 'prefab setup' again.")
            httpd.shutdown()

        timer = threading.Timer(args.timeout, timeout_handler)
        timer.start()

        try:
            with suppress(KeyboardInterrupt):
                httpd.serve_forever()
        finally:
            timer.cancel()
            httpd.server_close()
    else:
        print(f"Command {args.command} not recognized.")


if __name__ == "__main__":
    main()
