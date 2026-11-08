import threading
import time
from collections import OrderedDict

MAX_PRODUCTS = 5
TTL_SECONDS = 3600  # 1 jam

_STORE = OrderedDict()
_LOCK = threading.Lock()

def save_result(pid, **data):
    with _LOCK:
        _STORE[pid] = {**data, "created_at": time.time()}
        _STORE.move_to_end(pid)

        while len(_STORE) > MAX_PRODUCTS:
            _STORE.popitem(last=False)


def get_result(pid):
    with _LOCK:
        data = _STORE.get(pid)

        if data is None:
            return None

        if time.time() - data["created_at"] > TTL_SECONDS:
            del _STORE[pid]
            return None

        return data