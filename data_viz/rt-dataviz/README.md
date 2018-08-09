# Real-Time Data Visualization
This is a slightly modified version of the visualization created in [this tutorial](http://benjaminmbrown.github.io/2017-12-29-real-time-data-visualization/) (which has a link to the associated GitHub repo at the bottom).

The version here works with Python 3 and Tornado v4.5.3. It also dodges [this error](https://stackoverflow.com/questions/24851207/tornado-403-get-warning-when-opening-websocket) using the provided solution. The `check_origin` function is not a part of the code from the original blog post.

I also split `charts.js` off from the main `index.html` file simply because I prefer keeping my JS code separate from my HTML.

## To run
0. Use the `environment.yml` to set up a conda environment with the (few) requirements if necessary.
1. In one terminal, run `python websocket_server.py`
2. In another terminal, run `python -m http.server 3000`
3. Navigate to `localhost:3000`