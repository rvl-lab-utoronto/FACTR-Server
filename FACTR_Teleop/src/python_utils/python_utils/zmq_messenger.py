# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

import numpy as np
import threading
import zmq


class ZMQSubscriber:
    """
    Creates a thread that subscribes to a ZMQ publisher
    """
    def __init__(self, ip_address="tcp://192.168.1.3:2096", verbose=False):
        context = zmq.Context()
        self._sub_socket = context.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.CONFLATE, False)
        self._sub_socket.connect(ip_address)
        self._sub_socket.setsockopt(zmq.SUBSCRIBE, b'')

        self._subscriber_thread = threading.Thread(target=self._update_value)
        self._subscriber_thread.start()

        self._value = None
        self.verbose = verbose
        self.last_message = None

    @property
    def message(self):
        if self._value is None and self.verbose:
            print("The subscriber has not received a message")
        self.last_message = self._value
        return self._value
    
    def _update_value(self):
        while(True):
            message = self._sub_socket.recv()
            self._value = np.frombuffer(message).astype(np.float32)


class ZMQPublisher:
    """
    Creates a thread that publishes to a ZMQ subscriber
    """
    def __init__(self, ip_address="tcp://192.168.1.3:2096"):
        context = zmq.Context()
        self._pub_socket = context.socket(zmq.PUB)
        self._pub_socket.bind(ip_address)

        self.last_message = None

    def send_message(self, message):
        self.last_message = message
        self._pub_socket.send(message.astype(np.float64).tobytes())

    
