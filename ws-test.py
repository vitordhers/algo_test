# based on: https://github.com/bybit-exchange/pybit/blob/master/pybit/_http_manager.py

import hmac
import json
import time
import websocket
import datetime
import threading

api_key = "###"
api_secret = "###"


# Generate expires.
expires = int(time.time() * 10**3) + 5000

# Generate signature.
_val = f"GET/realtime{expires}"
signature = str(
    hmac.new(
        bytes(api_secret, "utf-8"),
        bytes(_val, "utf-8"),
        digestmod="sha256",
    ).hexdigest()
)


def on_open(ws):
    print("WebSocket connection opened.")
    auth_req = json.dumps({
        "op": "auth",
        "args": [api_key, str(expires), signature]
    })

    ws.send(auth_req)
    subscribe_req = json.dumps({
        "op": "subscribe",
        "args": [
            # "wallet",
            # "execution",
            # "position",
            "order"
        ]
    })
    ws.send(subscribe_req)


def on_message(ws, message):
    if '"op":"pong"' not in message:
        current_timestamp = time.time()
        print("TIMESTAMP = ", current_timestamp, message, ",", )


def on_error(ws, error):
    print("WebSocket error:", error)


def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed with status code:", close_status_code)


def send_heartbeat():
    time.sleep(20)
    while True:
        # Modify the heartbeat payload as per your requirement
        heartbeat_payload = json.dumps({"req_id": "100001", "op": "ping"})
        ws.send(heartbeat_payload)
        time.sleep(20)


url = "wss://stream-testnet.bybit.com/v5/private"

ws = websocket.WebSocketApp(
    url=url,
    on_open=on_open,
    on_message=on_message,
    on_close=on_close,
    on_error=on_error
)

# Create a separate thread for sending heartbeat packets
heartbeat_thread = threading.Thread(target=send_heartbeat)
# The thread will be terminated when the main thread (WebSocket) is closed
heartbeat_thread.daemon = True
heartbeat_thread.start()


ws.run_forever()
