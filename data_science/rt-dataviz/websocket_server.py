#! /usr/bin/env python
"""
Simple websocket server in Python.

Following this tutorial:
http://benjaminmbrown.github.io/2017-12-29-real-time-data-visualization/

Author: Jason Saporta
Date: 3/4/18
"""
import datetime
import json
import random
import time

import tornado
import tornado.websocket

payment_types = ["cash", "tab", "visa", "mastercard", "bitcoin"]
names_array = ["Ben", "Jarrod", "Vijay", "Aziz"]


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    """Control how data is sent to the client."""

    def check_origin(self, origin):
        """Avoid CORS-type problems."""
        return True

    def open(self):
        """
        Run when this socket is opened.

        The IOloop waits 3 seconds before starting to send data.
        """
        print("Connection established.")
        tornado.ioloop.IOLoop.instance().add_timeout(
            datetime.timedelta(seconds=3), self.send_data)

    def on_close(self):
        """Run when this socket is closed."""
        print("Connection closed.")

    def send_data(self):
        """
        Send new data for charts.

        This randomly generates data, packages it into a dictionary,
        and sends that data out as JSON.
        """
        print("Sending data.")

        qty = random.randrange(1, 4)
        total = random.randrange(30, 1000)
        tip = random.randrange(10, 100)
        pay_type = payment_types[random.randrange(0, 4)]
        name = names_array[random.randrange(0, 4)]
        spent = random.randrange(1, 150)
        year = random.randrange(2012, 2016)

        point_data = {"quantity": qty,
                      "total": total,
                      "tip": tip,
                      "pay_type": pay_type,
                      "Name": name,
                      "Spent": spent,
                      "Year": year,
                      "x": time.time()
                      }

        print(point_data)

        self.write_message(json.dumps(point_data))
        tornado.ioloop.IOLoop.instance().add_timeout(
            datetime.timedelta(seconds=1), self.send_data)


if __name__ == "__main__":
    print("Starting websocket server program.",
          "Awaiting client requests to open websocket...")
    application = tornado.web.Application([(r"/websocket",
                                            WebSocketHandler)])
    application.listen(8001)
    tornado.ioloop.IOLoop.instance().start()
